import React, { useState, useRef, useEffect, useCallback } from 'react';

// ─── Types ────────────────────────────────────────────────────────────────────

interface RunConfig {
  edited: string;
  raw: string;
  output: string;
  threshold: number;
  fps: number;
  buffer: number;
  device: string;
  noCut: boolean;
}

type Status = 'idle' | 'running' | 'done' | 'error';

// ─── helpers ─────────────────────────────────────────────────────────────────

const clean = (text: string) =>
  // strip ANSI escape codes
  text.replace(/\x1B\[[0-9;]*[mGKHF]/g, '');

// ─── Component ───────────────────────────────────────────────────────────────

export default function App() {
  const [config, setConfig] = useState<RunConfig>({
    edited: '',
    raw: '',
    output: '',
    threshold: 15,
    fps: 3,
    buffer: 0.5,
    device: 'auto',
    noCut: false,
  });

  const [showAdvanced, setShowAdvanced] = useState(false);
  const [status, setStatus]   = useState<Status>('idle');
  const [exitCode, setExitCode] = useState<number | null>(null);
  const [output, setOutput]   = useState('');
  const [previews, setPreviews] = useState<string[]>([]);
  const abortRef  = useRef<AbortController | null>(null);
  const termRef   = useRef<HTMLDivElement>(null);
  const isRunning = status === 'running';

  // auto-scroll terminal
  useEffect(() => {
    if (termRef.current) {
      termRef.current.scrollTop = termRef.current.scrollHeight;
    }
  }, [output]);

  const set = <K extends keyof RunConfig>(k: K, v: RunConfig[K]) =>
    setConfig(prev => ({ ...prev, [k]: v }));

  const append = useCallback((text: string) => {
    const cleaned = clean(text).replace(/\r\n/g, '\n').replace(/\r/g, '\n');
    setOutput(prev => prev + cleaned);
  }, []);

  const handleRun = async () => {
    if (isRunning) {
      abortRef.current?.abort();
      return;
    }

    setOutput('');
    setPreviews([]);
    setExitCode(null);
    setStatus('running');

    const ctrl = new AbortController();
    abortRef.current = ctrl;

    try {
      const res = await fetch('/api/run', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(config),
        signal:  ctrl.signal,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        append(`\n❌ Lỗi: ${err.error ?? res.statusText}\n`);
        setStatus('error');
        return;
      }

      const reader  = res.body!.getReader();
      const decoder = new TextDecoder();
      let buf = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buf += decoder.decode(value, { stream: true });

        // parse SSE lines
        const lines = buf.split('\n');
        buf = lines.pop() ?? '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          try {
            const { type, data } = JSON.parse(line.slice(6));
            if (type === 'out' || type === 'info') {
              const text = data as string;
              // Parse preview path
              const matches = text.match(/\[PREVIEW\]\s+(.+)/g);
              if (matches) {
                matches.forEach(m => {
                  const p = m.replace('[PREVIEW]', '').trim();
                  if (p) setPreviews(prev => [...prev, p]);
                });
              }
              append(text);
            } else if (type === 'done') {
              const { code, signal } = data as { code: number | null; signal: string | null };
              setExitCode(code ?? -1);
              setStatus(code === 0 ? 'done' : 'error');
              if (code !== null) {
                append(`\n─── Thoát với mã ${code} ───\n`);
              } else {
                append(`\n─── Bị dừng bởi ${signal} ───\n`);
              }
            } else if (type === 'error') {
              append(`\n❌ Lỗi server: ${data}\n`);
              setStatus('error');
            }
          } catch {}
        }
      }
    } catch (err: unknown) {
      if ((err as Error).name !== 'AbortError') {
        append(`\n❌ ${(err as Error).message}\n`);
        setStatus('error');
      } else {
        append('\n⚠️  Đã dừng.\n');
        setStatus('idle');
      }
    }
  };

  const btnLabel = isRunning ? '■ Dừng' : '▶ Chạy align.py';
  const btnClass = isRunning
    ? 'bg-red-600 hover:bg-red-500'
    : config.edited && config.raw
    ? 'bg-emerald-600 hover:bg-emerald-500'
    : 'bg-zinc-700 cursor-not-allowed opacity-50';

  const statusBadge: Record<Status, string> = {
    idle:    '',
    running: '🟡 Đang chạy...',
    done:    exitCode === 0 ? '🟢 Hoàn tất!' : '🔴 Lỗi',
    error:   '🔴 Lỗi',
  };

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 font-sans flex flex-col">
      {/* ── Header ── */}
      <header className="border-b border-zinc-800 px-8 py-5 flex items-center gap-3">
        <span className="text-2xl">🎬</span>
        <div>
          <h1 className="text-xl font-bold tracking-tight bg-gradient-to-r from-indigo-400 to-cyan-400 bg-clip-text text-transparent">
            Anime Recap Aligner
          </h1>
          <p className="text-xs text-zinc-500">Chạy align.py trực tiếp trên máy • CLIP + FFmpeg local</p>
        </div>
        {status !== 'idle' && (
          <span className="ml-auto text-sm font-medium">{statusBadge[status]}</span>
        )}
      </header>

      <div className="flex flex-1 overflow-hidden">
        {/* ── Config panel ── */}
        <aside className="w-80 shrink-0 border-r border-zinc-800 p-6 overflow-y-auto flex flex-col gap-5">

          {/* Edited path */}
          <Field label="📁 Video Edited" hint="Đường dẫn đầy đủ">
            <Input
              placeholder="D:\Videos\recap.mp4"
              value={config.edited}
              onChange={v => set('edited', v)}
              disabled={isRunning}
            />
          </Field>

          {/* Raw path */}
          <Field label="📂 Raw (Tập phim hoặc Thư mục)" hint="Dán link 1 file hoặc link nguyên 1 Thư mục">
            <Input
              placeholder="D:\Videos\Thu_Muc_Chua_Cac_Tap\"
              value={config.raw}
              onChange={v => set('raw', v)}
              disabled={isRunning}
            />
          </Field>

          {/* Advanced toggle */}
          <button
            onClick={() => setShowAdvanced(v => !v)}
            className="flex items-center gap-2 text-xs text-zinc-500 hover:text-zinc-300 transition-colors select-none"
          >
            <span className={`transition-transform ${showAdvanced ? 'rotate-90' : ''}`}>▶</span>
            Cài đặt nâng cao
          </button>

          {showAdvanced && (
            <div className="flex flex-col gap-4 pl-3 border-l border-zinc-800">
              {/* Output dir */}
              <Field label="📂 Thư mục output" hint="Bỏ trống = cuts/ cạnh file raw">
                <Input
                  placeholder="D:\cuts"
                  value={config.output}
                  onChange={v => set('output', v)}
                  disabled={isRunning}
                />
              </Field>

              {/* Threshold */}
              <Field label={`Ngưỡng khớp: ${config.threshold}%`} hint="Nhỏ = chặt hơn">
                <input
                  type="range" min={5} max={40} step={1}
                  value={config.threshold}
                  onChange={e => set('threshold', +e.target.value)}
                  disabled={isRunning}
                  className="w-full accent-indigo-500"
                />
              </Field>

              {/* FPS */}
              <Field label={`FPS phân tích: ${config.fps}`} hint="Frames/giây lấy mẫu">
                <input
                  type="range" min={1} max={6} step={1}
                  value={config.fps}
                  onChange={e => set('fps', +e.target.value)}
                  disabled={isRunning}
                  className="w-full accent-cyan-500"
                />
              </Field>

              {/* Buffer */}
              <Field label={`Buffer: ${config.buffer}s`} hint="Giây đệm đầu/cuối mỗi đoạn">
                <input
                  type="range" min={0} max={2} step={0.1}
                  value={config.buffer}
                  onChange={e => set('buffer', +e.target.value)}
                  disabled={isRunning}
                  className="w-full accent-violet-500"
                />
              </Field>

              {/* Device */}
              <Field label="Device">
                <select
                  value={config.device}
                  onChange={e => set('device', e.target.value)}
                  disabled={isRunning}
                  className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-indigo-500"
                >
                  <option value="auto">auto (tự detect)</option>
                  <option value="cuda">cuda (NVIDIA GPU)</option>
                  <option value="cpu">cpu</option>
                  <option value="mps">mps (Apple Silicon)</option>
                </select>
              </Field>

              {/* No-cut */}
              <label className="flex items-center gap-2 text-sm cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={config.noCut}
                  onChange={e => set('noCut', e.target.checked)}
                  disabled={isRunning}
                  className="accent-indigo-500 w-4 h-4"
                />
                <span className="text-zinc-300">--no-cut <span className="text-zinc-500">(chỉ xem kết quả)</span></span>
              </label>
            </div>
          )}

          {/* Run button */}
          <button
            onClick={handleRun}
            disabled={!isRunning && (!config.edited || !config.raw)}
            className={`mt-auto w-full py-3 rounded-xl font-bold text-base transition-colors flex items-center justify-center gap-2 ${btnClass}`}
          >
            {isRunning && <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />}
            {btnLabel}
          </button>
        </aside>

        {/* ── Terminal ── */}
        <main className="flex-1 flex flex-col bg-zinc-900 overflow-hidden">
          <div className="flex items-center gap-2 px-4 py-2 border-b border-zinc-800 text-xs text-zinc-500">
            <span className="w-3 h-3 rounded-full bg-red-500/70" />
            <span className="w-3 h-3 rounded-full bg-yellow-500/70" />
            <span className="w-3 h-3 rounded-full bg-green-500/70" />
            <span className="ml-2 font-mono">Terminal — align.py</span>
            {output && (
              <button
                onClick={() => setOutput('')}
                className="ml-auto hover:text-zinc-300 transition-colors"
              >
                xoá
              </button>
            )}
          </div>

          <div
            ref={termRef}
            className={`flex-1 overflow-y-auto p-4 font-mono text-sm leading-relaxed ${previews.length > 0 ? 'max-h-[50%]' : ''}`}
            style={{ background: '#0d0d0d' }}
          >
            {output ? (
              <pre className="whitespace-pre-wrap text-zinc-200">{output}</pre>
            ) : (
              <p className="text-zinc-600 italic mt-4 text-center">
                Nhập đường dẫn file và nhấn ▶ Chạy align.py
              </p>
            )}
            {isRunning && (
              <span className="inline-block w-2 h-4 bg-zinc-400 animate-pulse ml-1 align-middle" />
            )}
          </div>

          {/* ── Previews ── */}
          {previews.length > 0 && (
            <div className="flex-1 border-t border-zinc-800 bg-zinc-950 p-4 overflow-y-auto flex flex-col gap-4 w-full relative">
              <div className="sticky top-0 bg-zinc-950/90 backdrop-blur-sm pb-2 z-10 font-medium text-sm flex justify-between items-center px-1">
                <span>📸 Ảnh Preview ({previews.length})</span>
                <span className="text-xs text-zinc-500 font-normal">Edit vs Raw so khớp</span>
              </div>
              <div className="flex flex-col gap-6 items-center w-full max-w-5xl mx-auto">
                {previews.map((p, i) => (
                  <div key={i} className="w-full flex flex-col gap-2 bg-[#121212] p-3 rounded-xl border border-zinc-800 shadow-xl">
                    <img 
                      src={`/api/file?path=${encodeURIComponent(p)}&t=${Date.now()}`} 
                      alt={`Preview ${i + 1}`} 
                      className="w-full h-auto object-contain rounded bg-black max-h-[400px]" 
                      loading="lazy"
                    />
                  </div>
                ))}
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

// ─── Small UI components ──────────────────────────────────────────────────────

function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-1.5">
      <label className="text-xs font-semibold text-zinc-400 flex justify-between">
        <span>{label}</span>
        {hint && <span className="font-normal text-zinc-600">{hint}</span>}
      </label>
      {children}
    </div>
  );
}

function Input({
  placeholder, value, onChange, disabled,
}: {
  placeholder?: string;
  value: string;
  onChange: (v: string) => void;
  disabled?: boolean;
}) {
  return (
    <input
      type="text"
      placeholder={placeholder}
      value={value}
      onChange={e => onChange(e.target.value)}
      disabled={disabled}
      className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm font-mono focus:outline-none focus:border-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed placeholder:text-zinc-600"
    />
  );
}
