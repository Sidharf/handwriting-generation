import { useCallback, useEffect, useMemo, useRef, useState } from "react";

type Pair = {
  id: string;
  name: string;
  template_path: string;
  synthetic_path: string;
  source?: string;
};

type Style = {
  id: string;
  name: string;
  path: string;
  prepared_path?: string;
  style_text?: string;
};

type Cell = {
  id: string;
  page: number;
  bbox: number[];
  text: string;
  enabled: boolean;
};

type Job = {
  id: string;
  status: string;
  error?: string | null;
  result_pdf?: string | null;
};

type BucketSide = "template" | "synthetic";

type BucketState = {
  file: File | null;
  label: string | null;
  path: string | null;
  status: "empty" | "pending" | "saving" | "stored";
};

const emptyBucket = (): BucketState => ({
  file: null,
  label: null,
  path: null,
  status: "empty",
});

function basename(path: string): string {
  const parts = path.replace(/\\/g, "/").split("/");
  return parts[parts.length - 1] || path;
}

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  return res.json() as Promise<T>;
}

export default function App() {
  const [pairs, setPairs] = useState<Pair[]>([]);
  const [selected, setSelected] = useState<Pair | null>(null);
  const [styles, setStyles] = useState<Style[]>([]);
  const [activeStyleId, setActiveStyleId] = useState<string>("default");
  const [cells, setCells] = useState<Cell[]>([]);
  const [overlayUrl, setOverlayUrl] = useState<string | null>(null);
  const [maxNewTokens, setMaxNewTokens] = useState(128);
  const [seed, setSeed] = useState(0);
  const [variationMaster, setVariationMaster] = useState(35);
  const [variationAxes, setVariationAxes] = useState({
    diversity: 35,
    size: 35,
    placement: 35,
    stroke: 35,
  });
  const [variationManual, setVariationManual] = useState({
    diversity: false,
    size: false,
    placement: false,
    stroke: false,
  });
  const [showVariationAdvanced, setShowVariationAdvanced] = useState(false);
  const [job, setJob] = useState<Job | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragStyle, setDragStyle] = useState(false);
  const [dragBlank, setDragBlank] = useState(false);
  const [dragFilled, setDragFilled] = useState(false);
  const [styleTextDraft, setStyleTextDraft] = useState("");
  const [pendingStyleFile, setPendingStyleFile] = useState<File | null>(null);
  const [synthSamples, setSynthSamples] = useState<
    { id: string; png_base64: string; style_text: string; file: File }[]
  >([]);
  const synthFileById = useRef<Map<string, File>>(new Map());


  const [blank, setBlank] = useState<BucketState>(emptyBucket);
  const [filled, setFilled] = useState<BucketState>(emptyBucket);

  const refreshStyles = useCallback(async () => {
    const data = await api<{ styles: Style[]; active_style_id: string }>("/api/styles");
    setStyles(data.styles);
    setActiveStyleId(data.active_style_id);
  }, []);

  const refreshPairs = useCallback(async () => {
    const data = await api<{ pairs: Pair[] }>("/api/pairs/reference");
    setPairs(data.pairs);
  }, []);

  useEffect(() => {
    refreshPairs().catch((e) => setError(String(e)));
    refreshStyles().catch((e) => setError(String(e)));
  }, [refreshPairs, refreshStyles]);

  useEffect(() => {
    if (!job || job.status === "done" || job.status === "error") return;
    const t = setInterval(async () => {
      try {
        const j = await api<Job>(`/api/jobs/${job.id}`);
        setJob(j);
      } catch (e) {
        setError(String(e));
      }
    }, 2000);
    return () => clearInterval(t);
  }, [job]);

  const enabledCount = useMemo(
    () => cells.filter((c) => c.enabled).length,
    [cells]
  );

  const pairReady =
    blank.status === "stored" &&
    filled.status === "stored" &&
    Boolean(blank.path) &&
    Boolean(filled.path);

  async function detectFor(pair: Pair) {
    setBusy("detecting cells…");
    setError(null);
    setJob(null);
    try {
      const data = await api<{ cells: Cell[]; overlay_path?: string }>("/api/detect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          template_path: pair.template_path,
          synthetic_path: pair.synthetic_path,
          pair_id: pair.id,
          name: pair.name,
        }),
      });
      setCells(data.cells);
      setSelected(pair);
      const q = encodeURIComponent(JSON.stringify(data.cells));
      setOverlayUrl(
        `/api/overlay?template_path=${encodeURIComponent(pair.template_path)}&cells_json=${q}&page=0&t=${Date.now()}`
      );
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(null);
    }
  }

  function applyStoredPair(pair: Pair) {
    setBlank({
      file: null,
      label: basename(pair.template_path),
      path: pair.template_path,
      status: "stored",
    });
    setFilled({
      file: null,
      label: basename(pair.synthetic_path),
      path: pair.synthetic_path,
      status: "stored",
    });
    setSelected(pair);
    setCells([]);
    setOverlayUrl(null);
    setJob(null);
    setError(null);
  }

  async function uploadPairFiles(template: File, synthetic: File) {
    setBlank((b) => ({ ...b, status: "saving" }));
    setFilled((f) => ({ ...f, status: "saving" }));
    setBusy("saving pair locally…");
    setError(null);
    try {
      const fd = new FormData();
      fd.append("template", template);
      fd.append("synthetic", synthetic);
      fd.append("name", synthetic.name.replace(/\.pdf$/i, ""));
      const meta = await api<Pair>("/api/pairs/upload", { method: "POST", body: fd });
      setPairs((p) => [meta, ...p.filter((x) => x.id !== meta.id)]);
      setBlank({
        file: null,
        label: basename(meta.template_path),
        path: meta.template_path,
        status: "stored",
      });
      setFilled({
        file: null,
        label: basename(meta.synthetic_path),
        path: meta.synthetic_path,
        status: "stored",
      });
      setSelected(meta);
      setCells([]);
      setOverlayUrl(null);
      setJob(null);
    } catch (e) {
      setBlank((b) => ({
        ...b,
        status: b.file ? "pending" : b.path ? "stored" : "empty",
      }));
      setFilled((f) => ({
        ...f,
        status: f.file ? "pending" : f.path ? "stored" : "empty",
      }));
      setError(String(e));
    } finally {
      setBusy(null);
    }
  }

  async function stageBucket(side: BucketSide, file: File | undefined) {
    if (!file || !/\.pdf$/i.test(file.name)) {
      setError("Drop a single PDF into that bucket");
      return;
    }
    setError(null);
    setCells([]);
    setOverlayUrl(null);
    setJob(null);

    const next: BucketState = {
      file,
      label: file.name,
      path: null,
      status: "pending",
    };

    const other = side === "template" ? filled : blank;
    if (side === "template") setBlank(next);
    else setFilled(next);

    // If the other side already has a File, upload both.
    // If the other side is stored-from-disk only (no File), wait until user also
    // drops a File into this side and we have two Files — or they pick a saved pair.
    if (other.file) {
      const template = side === "template" ? file : other.file;
      const synthetic = side === "synthetic" ? file : other.file;
      await uploadPairFiles(template, synthetic);
    } else if (other.path && other.status === "stored") {
      // Other side is a stored path without a File — can't re-upload without bytes.
      // Keep pending until user clears the other side and re-adds, or picks a saved pair.
      // If both were File-based we're fine; for mixed, require both Files or use saved pairs.
    }
  }

  function clearBucket(side: BucketSide) {
    if (side === "template") setBlank(emptyBucket());
    else setFilled(emptyBucket());
    setSelected(null);
    setCells([]);
    setOverlayUrl(null);
    setJob(null);
  }

  function onDetectClick() {
    if (!pairReady || !blank.path || !filled.path) return;
    const pair: Pair = selected ?? {
      id: "adhoc",
      name: filled.label || blank.label || "pair",
      template_path: blank.path,
      synthetic_path: filled.path,
      source: "upload",
    };
    void detectFor({
      ...pair,
      template_path: blank.path,
      synthetic_path: filled.path,
    });
  }

  async function uploadStyle(file: File, styleText: string) {
    const trimmed = styleText.trim();
    if (!trimmed) {
      setError("Type the exact transcription of the handwriting in the style PNG");
      return;
    }
    setBusy("uploading style…");
    setError(null);
    try {
      const fd = new FormData();
      fd.append("file", file);
      fd.append("style_text", trimmed);
      await api("/api/styles/upload", { method: "POST", body: fd });
      setPendingStyleFile(null);
      setStyleTextDraft("");
      await refreshStyles();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(null);
    }
  }

  function onStyleFiles(files: FileList | File[], opts?: { styleText?: string }) {
    const file = Array.from(files).find(
      (f) => /\.png$/i.test(f.name) || f.type === "image/png"
    );
    if (!file) {
      setError("Drop a PNG handwriting sample");
      return;
    }
    setPendingStyleFile(file);
    if (opts?.styleText) {
      setStyleTextDraft(opts.styleText);
    } else if (/^synth-biotech/i.test(file.name)) {
      setStyleTextDraft("biotech is cool");
    }
    setError(null);
  }

  async function generateSynthStyles() {
    setBusy("generating 4 handwriting samples…");
    setError(null);
    try {
      const data = await api<{
        images: { id: string; png_base64: string; style_text: string; index?: number }[];
        style_text: string;
      }>("/api/styles/synthesize", { method: "POST" });
      const next: { id: string; png_base64: string; style_text: string; file: File }[] = [];
      const map = new Map<string, File>();
      for (const img of data.images) {
        const bin = Uint8Array.from(atob(img.png_base64), (c) => c.charCodeAt(0));
        const file = new File([bin], `synth-biotech-${img.index ?? next.length}.png`, {
          type: "image/png",
        });
        map.set(img.id, file);
        next.push({
          id: img.id,
          png_base64: img.png_base64,
          style_text: img.style_text || data.style_text || "biotech is cool",
          file,
        });
      }
      synthFileById.current = map;
      setSynthSamples(next);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(null);
    }
  }

  async function activateStyle(id: string) {
    await api(`/api/styles/${id}/activate`, { method: "POST" });
    setActiveStyleId(id);
  }

  async function saveStyleText(id: string, styleText: string) {
    const trimmed = styleText.trim();
    if (!trimmed) {
      setError("style_text cannot be empty");
      return;
    }
    setBusy("saving style transcription…");
    setError(null);
    try {
      await api(`/api/styles/${id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ style_text: trimmed }),
      });
      await refreshStyles();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(null);
    }
  }

  function updateCell(id: string, patch: Partial<Cell>) {
    setCells((prev) => {
      const next = prev.map((c) => (c.id === id ? { ...c, ...patch } : c));
      if (selected || blank.path) {
        const q = encodeURIComponent(JSON.stringify(next));
        const tPath = selected?.template_path || blank.path!;
        setOverlayUrl(
          `/api/overlay?template_path=${encodeURIComponent(tPath)}&cells_json=${q}&page=0&t=${Date.now()}`
        );
      }
      return next;
    });
  }

  async function runJob() {
    if (!pairReady || !blank.path || !filled.path || enabledCount === 0) return;
    setBusy("starting job…");
    setError(null);
    try {
      const j = await api<Job>("/api/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          template_path: blank.path,
          synthetic_path: filled.path,
          cells,
          max_new_tokens: maxNewTokens,
          seed,
          variation: {
            master: variationMaster,
            diversity: variationAxes.diversity,
            size: variationAxes.size,
            placement: variationAxes.placement,
            stroke: variationAxes.stroke,
          },
          pair_id: selected?.id,
          pair_name: selected?.name || filled.label || "pair",
        }),
      });
      setJob(j);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(null);
    }
  }

  function chipFor(b: BucketState) {
    if (b.status === "empty") return <span className="chip empty">Not set</span>;
    if (b.status === "saving") return <span className="chip saving">Saving…</span>;
    if (b.status === "pending") return <span className="chip pending">Ready (not saved yet)</span>;
    return <span className="chip stored">Stored locally</span>;
  }

  function renderBucket(
    side: BucketSide,
    title: string,
    hint: string,
    state: BucketState,
    dragging: boolean,
    setDragging: (v: boolean) => void
  ) {
    return (
      <div className={`bucket ${dragging ? "drag" : ""} ${state.status}`}>
        <div className="bucket-head">
          <strong>{title}</strong>
          {chipFor(state)}
        </div>
        <p className="bucket-hint">{hint}</p>
        {state.label ? (
          <div className="bucket-file">
            <span className="bucket-name" title={state.path || state.label}>
              {state.label}
            </span>
            <button
              type="button"
              className="ghost clear"
              aria-label={`Clear ${title}`}
              onClick={() => clearBucket(side)}
            >
              ×
            </button>
          </div>
        ) : (
          <div
            className="bucket-drop"
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              const f = e.dataTransfer.files?.[0];
              void stageBucket(side, f);
            }}
          >
            Drop PDF here
            <div className="actions" style={{ justifyContent: "center", marginTop: 8 }}>
              <label className="btn secondary">
                Browse
                <input
                  type="file"
                  accept="application/pdf"
                  hidden
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    void stageBucket(side, f);
                    e.target.value = "";
                  }}
                />
              </label>
            </div>
          </div>
        )}
        {state.status === "pending" && !state.path && (
          <p className="bucket-note">Add the other PDF to save both under data/pairs/</p>
        )}
      </div>
    );
  }

  return (
    <div className="app">
      <header className="top">
        <div>
          <h1>Handwriting Generation</h1>
          <p>
            Diff template vs synthetic PDFs, batch-generate styled ink on Modal (L4),
            stamp blue handwriting onto the template.
          </p>
        </div>
        <div className="badge">Emuru · Modal · ink #2563EB</div>
      </header>

      {error && <p className="status err">{error}</p>}
      {busy && <p className="status warn">{busy}</p>}

      <div className="grid">
        <div className="stack">
          <section className="panel">
            <h2>1 · Document pair</h2>
            <p className="sub">
              Add a blank template and a filled synthetic PDF. Files save under{" "}
              <code>data/pairs/</code> when both are set.
            </p>

            <div className="buckets">
              {renderBucket(
                "template",
                "Blank (template)",
                "Empty form PDF",
                blank,
                dragBlank,
                setDragBlank
              )}
              {renderBucket(
                "synthetic",
                "Filled (synthetic)",
                "Typed / filled PDF",
                filled,
                dragFilled,
                setDragFilled
              )}
            </div>

            <div className="pair-ready">
              {pairReady ? (
                <>
                  <div className="pair-ready-meta">
                    <span className="chip stored">Pair ready</span>
                    <span className="pair-ready-name">
                      {selected?.name || filled.label || "Document pair"}
                    </span>
                  </div>
                  <button
                    className="btn accent"
                    disabled={Boolean(busy)}
                    onClick={onDetectClick}
                  >
                    Detect fills
                  </button>
                </>
              ) : (
                <p className="sub" style={{ margin: 0 }}>
                  {blank.status === "empty" && filled.status === "empty"
                    ? "Drop a PDF into each bucket, or pick a saved pair below."
                    : "Add the other PDF to continue."}
                </p>
              )}
            </div>

            <h3 className="list-title">Saved pairs</h3>
            <div className="list">
              {pairs.map((p) => (
                <div
                  key={p.id}
                  className={`row ${selected?.id === p.id ? "active" : ""}`}
                >
                  <div className="meta">
                    <strong>{p.name}</strong>
                    <div style={{ color: "var(--muted)", fontSize: "0.75rem" }}>
                      {p.source || "upload"} · already on disk
                    </div>
                  </div>
                  <button
                    className="ghost"
                    onClick={() => applyStoredPair(p)}
                  >
                    Use pair
                  </button>
                </div>
              ))}
              {pairs.length === 0 && (
                <div className="row">
                  <span className="meta">No saved pairs yet</span>
                </div>
              )}
            </div>
          </section>

          <section className="panel">
            <h2>2 · Handwriting samples</h2>
            <p className="sub">
              Drop a PNG style sample, then type the <strong>exact transcription</strong> of
              the writing in that image (Emuru needs both). Prefer a single line of{" "}
              <strong>IDs / dates / digits</strong> (e.g.{" "}
              <code>DS-26052 TN 10JUN26</code>), not English sentences. Active sample
              conditions the whole batch.
            </p>
            <div
              className={`drop ${dragStyle ? "drag" : ""}`}
              onDragOver={(e) => {
                e.preventDefault();
                setDragStyle(true);
              }}
              onDragLeave={() => setDragStyle(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragStyle(false);
                const synthId = e.dataTransfer.getData("application/x-handwriting-synth");
                if (e.dataTransfer.files?.length) {
                  onStyleFiles(e.dataTransfer.files, {
                    styleText: synthId ? "biotech is cool" : undefined,
                  });
                  return;
                }
                if (synthId && synthFileById.current.has(synthId)) {
                  onStyleFiles([synthFileById.current.get(synthId)!], {
                    styleText: "biotech is cool",
                  });
                }
              }}
            >
              Drop handwriting PNG here
              <div className="actions" style={{ justifyContent: "center" }}>
                <label className="btn secondary">
                  Browse PNG
                  <input
                    type="file"
                    accept="image/png"
                    hidden
                    onChange={(e) => e.target.files && onStyleFiles(e.target.files)}
                  />
                </label>
                <button
                  type="button"
                  className="btn accent"
                  disabled={Boolean(busy)}
                  onClick={() => void generateSynthStyles()}
                >
                  Generate 4 samples
                </button>
              </div>
            </div>

            {synthSamples.length > 0 && (
              <div className="synth-gallery">
                <p className="sub">
                  Drag one sample into the drop zone above (transcription:{" "}
                  <code>biotech is cool</code>).
                </p>
                <div className="synth-grid">
                  {synthSamples.map((s, i) => (
                    <button
                      type="button"
                      key={s.id}
                      className="synth-tile"
                      title="Drag into the drop zone, or click to load"
                      draggable
                      onDragStart={(e) => {
                        e.dataTransfer.effectAllowed = "copy";
                        e.dataTransfer.setData("application/x-handwriting-synth", s.id);
                        e.dataTransfer.setData("text/plain", s.id);
                        try {
                          e.dataTransfer.items.add(s.file);
                        } catch {
                          /* some browsers reject File on items.add */
                        }
                      }}
                      onClick={() =>
                        onStyleFiles([s.file], { styleText: s.style_text || "biotech is cool" })
                      }
                    >
                      <img
                        src={`data:image/png;base64,${s.png_base64}`}
                        alt={`Synthetic sample ${i + 1}`}
                        draggable={false}
                      />
                      <span>Sample {i + 1}</span>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {pendingStyleFile && (
              <div className="style-upload-form">
                <p className="sub" style={{ marginBottom: 8 }}>
                  Pending: <strong>{pendingStyleFile.name}</strong> — enter transcription, then upload.
                </p>
                <div className="field" style={{ flex: 1 }}>
                  <label>Style transcription</label>
                  <input
                    type="text"
                    value={styleTextDraft}
                    placeholder='e.g. DS-26052 TN 10JUN26'
                    onChange={(e) => setStyleTextDraft(e.target.value)}
                  />
                </div>
                <div className="actions">
                  <button
                    className="btn accent"
                    disabled={Boolean(busy) || !styleTextDraft.trim()}
                    onClick={() => void uploadStyle(pendingStyleFile, styleTextDraft)}
                  >
                    Upload style
                  </button>
                  <button
                    className="btn secondary"
                    type="button"
                    onClick={() => {
                      setPendingStyleFile(null);
                      setStyleTextDraft("");
                    }}
                  >
                    Cancel
                  </button>
                </div>
              </div>
            )}

            <div className="list">
              {styles.map((s) => (
                <div
                  key={s.id}
                  className={`row ${activeStyleId === s.id ? "active" : ""}`}
                >
                  <div style={{ display: "flex", gap: 10, alignItems: "center", flex: 1, minWidth: 0 }}>
                    <img
                      className="thumb"
                      src={`/api/styles/${s.id}/image?prepared=1`}
                      alt={s.name}
                      onError={(e) => {
                        (e.target as HTMLImageElement).src = `/api/styles/${s.id}/image`;
                      }}
                    />
                    <div className="meta" style={{ flex: 1, minWidth: 0 }}>
                      <div>{s.name}</div>
                      <input
                        className="style-text-input"
                        type="text"
                        defaultValue={s.style_text || ""}
                        key={`${s.id}-${s.style_text || ""}`}
                        placeholder="Transcription of this PNG"
                        title="Must match the glyphs in the style image"
                        onBlur={(e) => {
                          const next = e.target.value.trim();
                          if (next && next !== (s.style_text || "").trim()) {
                            void saveStyleText(s.id, next);
                          }
                        }}
                      />
                    </div>
                  </div>
                  <button
                    className="ghost"
                    disabled={activeStyleId === s.id}
                    onClick={() => activateStyle(s.id)}
                  >
                    {activeStyleId === s.id ? "Active" : "Use"}
                  </button>
                </div>
              ))}
            </div>
          </section>
        </div>

        <div className="stack">
          <section className="panel">
            <h2>3 · Detected cells & generate</h2>
            <p className="sub">
              Typed values on the filled PDF (black or blue) that are not on the
              template become handwriting cells, then stamped onto the blank.
              Edit text or uncheck before running.
            </p>

            {overlayUrl && (
              <div className="preview">
                <img src={overlayUrl} alt="Detected fill overlay" />
              </div>
            )}

            {cells.length > 0 ? (
              <div style={{ maxHeight: 260, overflow: "auto", marginTop: 12 }}>
                <table className="cells">
                  <thead>
                    <tr>
                      <th></th>
                      <th>Page</th>
                      <th>Text</th>
                    </tr>
                  </thead>
                  <tbody>
                    {cells.map((c) => (
                      <tr key={c.id}>
                        <td>
                          <input
                            type="checkbox"
                            checked={c.enabled}
                            onChange={(e) =>
                              updateCell(c.id, { enabled: e.target.checked })
                            }
                          />
                        </td>
                        <td>{c.page + 1}</td>
                        <td>
                          <input
                            type="text"
                            value={c.text}
                            onChange={(e) => updateCell(c.id, { text: e.target.value })}
                          />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="sub">Load both PDFs and run Detect fills to see cells.</p>
            )}

            <div className="actions" style={{ marginTop: 14 }}>
              <div className="field">
                <label>Max new tokens</label>
                <input
                  type="number"
                  min={16}
                  max={256}
                  value={maxNewTokens}
                  onChange={(e) => setMaxNewTokens(Number(e.target.value))}
                />
              </div>
              <div className="field">
                <label>Seed</label>
                <input
                  type="number"
                  value={seed}
                  onChange={(e) => setSeed(Number(e.target.value))}
                />
              </div>
            </div>

            <div className="variation-block">
              <div className="field field-wide">
                <label>
                  Cell variation —{" "}
                  {variationMaster <= 0
                    ? "None"
                    : variationMaster < 30
                      ? "Subtle"
                      : variationMaster < 65
                        ? "Natural"
                        : "Wild"}{" "}
                  ({variationMaster})
                </label>
                <input
                  type="range"
                  min={0}
                  max={100}
                  value={variationMaster}
                  onChange={(e) => {
                    const v = Number(e.target.value);
                    setVariationMaster(v);
                    setVariationAxes((prev) => ({
                      diversity: variationManual.diversity ? prev.diversity : v,
                      size: variationManual.size ? prev.size : v,
                      placement: variationManual.placement ? prev.placement : v,
                      stroke: variationManual.stroke ? prev.stroke : v,
                    }));
                  }}
                />
                <p className="sub variation-hint">
                  Increases cell-to-cell differences; Seed still controls the base look.
                </p>
              </div>
              <button
                type="button"
                className="btn ghost"
                onClick={() => setShowVariationAdvanced((s) => !s)}
              >
                {showVariationAdvanced ? "Hide advanced" : "Advanced variation"}
              </button>
              {showVariationAdvanced && (
                <div className="variation-advanced">
                  {(
                    [
                      ["diversity", "Diversity (glyph / seed)"],
                      ["size", "Size"],
                      ["placement", "Placement + slant"],
                      ["stroke", "Stroke / ink"],
                    ] as const
                  ).map(([key, label]) => (
                    <div className="field field-wide" key={key}>
                      <label>
                        {label} ({variationAxes[key]})
                        {variationManual[key] ? " · manual" : ""}
                      </label>
                      <input
                        type="range"
                        min={0}
                        max={100}
                        value={variationAxes[key]}
                        onChange={(e) => {
                          const v = Number(e.target.value);
                          setVariationManual((m) => ({ ...m, [key]: true }));
                          setVariationAxes((prev) => ({ ...prev, [key]: v }));
                        }}
                      />
                    </div>
                  ))}
                  <button
                    type="button"
                    className="btn"
                    onClick={() => {
                      setVariationManual({
                        diversity: false,
                        size: false,
                        placement: false,
                        stroke: false,
                      });
                      setVariationAxes({
                        diversity: variationMaster,
                        size: variationMaster,
                        placement: variationMaster,
                        stroke: variationMaster,
                      });
                    }}
                  >
                    Reset axes to master
                  </button>
                </div>
              )}
            </div>

            <div className="actions">
              <button
                className="btn accent"
                disabled={!pairReady || enabledCount === 0 || Boolean(busy)}
                onClick={runJob}
              >
                Run batch ({enabledCount} cells)
              </button>
            </div>

            {job && (
              <div className="status">
                Job {job.id}: <strong>{job.status}</strong>
                {job.status === "done" && job.id && (
                  <>
                    {" · "}
                    <a href={`/api/jobs/${job.id}/result.pdf`} target="_blank" rel="noreferrer">
                      Download PDF
                    </a>
                  </>
                )}
                {job.status === "error" && job.error && (
                  <pre style={{ whiteSpace: "pre-wrap", color: "var(--err)" }}>
                    {job.error.slice(0, 800)}
                  </pre>
                )}
              </div>
            )}
          </section>
        </div>
      </div>
    </div>
  );
}
