import argparse
import asyncio
import sqlite3
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8001)
parser.add_argument("--db", type=str, default="bot_database.db")
args, _ = parser.parse_known_args()

DB_FILE = args.db
PORT = args.port

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
            html_content = f.read()
        html_content = html_content.replace("{{SYMBOL}}", f"XAUUSDm (Quant Engine Port {PORT})")
        html_content = html_content.replace("{{ACCOUNT_TYPE}}", "PPO + FinBERT SUPERVISOR")
        return HTMLResponse(html_content)
    except Exception as e:
        return HTMLResponse(f"<h1>Error reading index.html: {e}</h1>")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    async def receive_messages():
        while True:
            try:
                data = await websocket.receive_json()
                if "command" in data:
                    new_status = "RUNNING" if data["command"] == "start_bot" else "PAUSED"
                    with sqlite3.connect(DB_FILE) as conn:
                        conn.cursor().execute("UPDATE bot_control SET status=? WHERE id=1", (new_status,))
                        conn.commit()
            except Exception:
                await asyncio.sleep(1)

    async def send_updates():
        last_log_id = 0
        while True:
            try:
                with sqlite3.connect(DB_FILE) as conn:
                    c = conn.cursor()

                    if last_log_id == 0:
                        try:
                            c.execute("SELECT MAX(id) FROM bot_logs")
                            row = c.fetchone()
                            if row and row[0]:
                                last_log_id = row[0]
                        except: pass

                    try:
                        c.execute("SELECT balance, profit, position, bid, ask FROM current_state WHERE id=1")
                        state = c.fetchone()

                        c.execute("SELECT bias, reasoning, last_update FROM gemini_bias WHERE id=1")
                        gemini_data = c.fetchone()

                        if state:
                            balance, profit, position, bid, ask = state
                            g_bias = gemini_data[0] if gemini_data else "NEUTRAL"
                            g_reason = gemini_data[1] if gemini_data else "Waiting for FinBERT..."

                            await websocket.send_json({
                                "type": "state",
                                "balance": f"${balance:,.2f}",
                                "profit": f"{'+' if profit >= 0 else '-'}${abs(profit):,.2f}",
                                "position": position,
                                "bid": bid,
                                "ask": ask,
                                "gemini_bias": g_bias,
                                "gemini_reasoning": g_reason
                            })
                    except Exception:
                        pass 

                    try:
                        c.execute("SELECT id, timestamp, log_type, message FROM bot_logs WHERE id > ?", (last_log_id,))
                        logs = c.fetchall()
                        for log in logs:
                            log_id, ts, log_type, msg = log
                            time_str = ts.split(" ")[1] if " " in ts else ts
                            await websocket.send_json({
                                "type": "log",
                                "log_type": log_type,
                                "message": msg,
                                "time": time_str
                            })
                            last_log_id = max(last_log_id, log_id)
                    except Exception:
                        pass

                    try:
                        c.execute("SELECT timestamp_close, action, lot_size, profit, exit_reason FROM trade_details WHERE status='CLOSED' ORDER BY id DESC")
                        trades = c.fetchall()
                        trades_data = []
                        for t in trades:
                            time_val = t[0].split(" ")[1] if t[0] and " " in t[0] else "-"
                            trades_data.append({
                                "time": time_val,
                                "action": t[1],
                                "lot": t[2],
                                "profit": t[3],
                                "reason": t[4]
                            })
                        await websocket.send_json({
                            "type": "trades",
                            "data": trades_data
                        })
                    except Exception:
                        pass
            except Exception:
                pass
            await asyncio.sleep(0.5)

    task1 = asyncio.create_task(receive_messages())
    task2 = asyncio.create_task(send_updates())

    try:
        await asyncio.wait([task1, task2], return_when=asyncio.FIRST_COMPLETED)
    except WebSocketDisconnect:
        pass
    finally:
        task1.cancel()
        task2.cancel()

if __name__ == "__main__":
    print("="*50)
    print(f"🚀 Mulai Server Dashboard V2 Hybrid 🚀")
    print(f"📡 Port      : {PORT}")
    print(f"🗄️ Database  : {DB_FILE}")
    print(f"🌐 Akses Web : http://0.0.0.0:{PORT}")
    print("="*50)
    uvicorn.run(app, host="0.0.0.0", port=PORT)