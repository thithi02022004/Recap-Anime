import express from 'express';
import { spawn, ChildProcess } from 'child_process';
import { fileURLToPath } from 'url';
import path from 'path';
import fs from 'fs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PYTHON = process.platform === 'win32' ? 'python' : 'python3';
const ALIGN_PY = path.join(__dirname, 'align.py');

const app = express();
app.use(express.json());

app.get('/api/health', (_req, res) => {
  res.json({ ok: true, alignPy: ALIGN_PY });
});

app.get('/api/file', (req, res) => {
  const file = req.query.path as string;
  if (!file) {
    res.status(400).send('Missing path');
    return;
  }
  res.sendFile(path.resolve(file));
});

app.post('/api/run', (req, res) => {
  const { edited, raw, output, threshold, fps, buffer, device, noCut } = req.body;

  if (!edited?.trim() || !raw?.trim()) {
    res.status(400).json({ error: 'Thiếu đường dẫn file edited hoặc raw.' });
    return;
  }

  const rawInput = raw.replace(/^"|"$/g, '').trim();
  let rawFiles: string[] = [];

  try {
    const stat = fs.statSync(rawInput);
    if (stat.isDirectory()) {
      // Quét toàn bộ video trong folder
      const files = fs.readdirSync(rawInput);
      const videoExts = ['.mp4', '.mkv', '.webm', '.avi', '.ts', '.mov'];
      rawFiles = files
        .filter(f => videoExts.includes(path.extname(f).toLowerCase()))
        .sort((a, b) => a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' }))
        .map(f => path.join(rawInput, f));
      
      if (rawFiles.length === 0) {
        res.status(400).json({ error: `Không tìm thấy file video nào trong thư mục: ${rawInput}` });
        return;
      }
    } else {
      rawFiles = [rawInput]; // Là một file duy nhất
    }
  } catch (err) {
    // Nếu file chưa tồn tại hoặc lỗi đường dẫn
    rawFiles = [rawInput];
  }

  const args: string[] = [
    '-u',          // unbuffered stdout – bắt buộc để stream real-time
    ALIGN_PY,
    '--edited', edited.replace(/^"|"$/g, '').trim(),
    '--raw',    ...rawFiles,
    '--threshold', String(threshold ?? 15),
    '--fps',       String(fps ?? 3),
    '--buffer',    String(buffer ?? 0.5),
  ];
  if (output?.trim())              args.push('--output', output.replace(/^"|"$/g, '').trim());
  if (device && device !== 'auto') args.push('--device', device);
  if (noCut)                       args.push('--no-cut');

  // Server-Sent Events
  res.setHeader('Content-Type',  'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection',    'keep-alive');
  res.flushHeaders();

  const send = (type: string, data: unknown) => {
    try { res.write(`data: ${JSON.stringify({ type, data })}\n\n`); } catch {}
  };

  send('info', `▶ ${PYTHON} ${args.slice(1).join(' ')}\n`);

  const proc: ChildProcess = spawn(PYTHON, args, {
    cwd: __dirname,
    stdio: ['ignore', 'pipe', 'pipe'],
    env: { ...process.env, PYTHONIOENCODING: 'utf-8' },
  });

  proc.stdout?.on('data', (d: Buffer) => send('out', d.toString('utf-8')));
  proc.stderr?.on('data', (d: Buffer) => send('out', d.toString('utf-8'))); // tqdm writes to stderr
  proc.on('close',  (code, signal) => { 
    send('done', { code, signal }); 
    res.end(); 
  });
  proc.on('error',  (err)  => { send('error', err.message); res.end(); });

  // Lắng nghe res.on('close') để huỷ process nếu client ngắt ngầm (vd bấn Dừng)
  res.on('close', () => {
    // Nếu chưa gọi res.end() mà connection bị đóng tức là client disconnect
    if (!res.writableEnded) {
      try { proc.kill(); } catch {}
    }
  });
});

const PORT = 3001;
app.listen(PORT, () => console.log(`🎬  Backend: http://localhost:${PORT}`));
