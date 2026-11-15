import argparse
import asyncio
import json
import sqlite3
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8001)
parser.add_argument("--db", type=str, default="bot_database.db")
parser.add_argument("--symbol", type=str, default="XAUUSDm", help="Label simbol di header")
parser.add_argument("--mode", type=str, choices=["demo", "real"], default="demo", help="Label mode di header")
parser.add_argument("--currency", type=str, default="$", help="Satuan saldo/profit, mis. USC untuk akun cent")
args, _ = parser.parse_known_args()

DB_FILE = args.db
PORT = args.port
CURRENCY = args.currency

app = FastAPI()

def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS bot_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, log_type TEXT, message TEXT)")
        c.execute("CREATE TABLE IF NOT EXISTS bot_control (id INTEGER PRIMARY KEY CHECK (id = 1), status TEXT)")
        c.execute("INSERT OR IGNORE INTO bot_control (id, status) VALUES (1, 'PAUSED')")
        conn.commit()

init_db()

@app.get("/")
async def get():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            html = f.read()
        html = html.replace("{{SYMBOL}}", args.symbol)
        html = html.replace("{{MODE}}", args.mode)
        html = html.replace("{{CURRENCY}}", CURRENCY)
        html = html.replace("{{PORT}}", str(PORT))
        return HTMLResponse(html)
    except Exception as e:
        return HTMLResponse(f"<h1>Error reading index.html: {e}</h1>")

def set_status(command):
    new_status = "RUNNING" if command == "start_bot" else "PAUSED"
    with sqlite3.connect(DB_FILE) as conn:
        conn.cursor().execute("UPDATE bot_control SET status=? WHERE id=1", (new_status,))
        conn.commit()

def read_state():
    out = {"type": "state", "currency": CURRENCY, "bot_status": "PAUSED"}
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        try:
            row = c.execute("SELECT status FROM bot_control WHERE id=1").fetchone()
            if row: out["bot_status"] = row[0]
        except Exception:
            pass
        try:
            row = c.execute("SELECT balance, profit, position, bid, ask FROM current_state WHERE id=1").fetchone()
            if row:
                out.update(balance=row[0], profit=row[1], position=row[2], bid=row[3], ask=row[4])
        except Exception:
            pass
        try:
            row = c.execute("SELECT bias, reasoning, last_update FROM gemini_bias WHERE id=1").fetchone()
            if row:
                out.update(gemini_bias=row[0], gemini_reasoning=row[1], gemini_time=row[2])
        except Exception:
            pass
    return out

def read_new_logs(last_id, history=False):
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        try:
            if history:
                rows = c.execute("SELECT id, timestamp, log_type, message FROM (SELECT * FROM bot_logs ORDER BY id DESC LIMIT 40) ORDER BY id ASC").fetchall()
            else:
                rows = c.execute("SELECT id, timestamp, log_type, message FROM bot_logs WHERE id > ? ORDER BY id ASC", (last_id,)).fetchall()
        except Exception:
            return last_id, []
    msgs = []
    for log_id, ts, log_type, msg in rows:
        msgs.append({"type": "log", "log_type": log_type, "message": msg,
                     "time": ts.split(" ")[1] if ts and " " in ts else ts})
        last_id = max(last_id, log_id)
    return last_id, msgs

def read_trades():
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        try:
            agg = c.execute("SELECT COUNT(*), COALESCE(SUM(profit > 0), 0), COALESCE(SUM(profit), 0) FROM trade_details WHERE status='CLOSED'").fetchone()
            rows = c.execute("SELECT timestamp_close, action, lot_size, profit, exit_reason FROM trade_details WHERE status='CLOSED' ORDER BY id DESC LIMIT 50").fetchall()
        except Exception:
            return None
    return {"type": "trades",
            "stats": {"count": agg[0], "wins": agg[1], "total": agg[2]},
            "data": [{"ts": r[0] or "", "action": r[1], "lot": r[2], "profit": r[3] or 0.0, "reason": r[4] or ""} for r in rows]}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    async def receive_messages():
        while True:
            try:
                data = await websocket.receive_json()
            except (WebSocketDisconnect, RuntimeError):
                return
            except Exception:
                continue
            if isinstance(data, dict) and "command" in data:
                try:
                    set_status(data["command"])
                except Exception:
                    pass

    async def send_updates():
        last_log_id = 0
        first = True
        last_trades = None
        while True:
            try:
                last_log_id, logs = read_new_logs(last_log_id, history=first)
                first = False
                for m in logs:
                    await websocket.send_json(m)
                await websocket.send_json(read_state())
                trades = read_trades()
                if trades is not None:
                    dump = json.dumps(trades, sort_keys=True)
                    if dump != last_trades:
                        last_trades = dump
                        await websocket.send_json(trades)
            except (WebSocketDisconnect, RuntimeError):
                return
            except Exception:
                pass
            await asyncio.sleep(0.5)

    tasks = [asyncio.create_task(receive_messages()), asyncio.create_task(send_updates())]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for t in tasks:
            t.cancel()

if __name__ == "__main__":
    print("=" * 50)
    print("Dashboard Quant Engine")
    print(f"Port      : {PORT}")
    print(f"Database  : {DB_FILE}")
    print(f"Mode      : {args.mode.upper()} | {args.symbol} | {CURRENCY}")
    print(f"Akses Web : http://127.0.0.1:{PORT}")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=PORT)
