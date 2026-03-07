'use client';

import React, { useState, useEffect, useRef } from 'react';
import { X, Cpu, DownloadCloud, Database, RefreshCcw, TrendingUp, Trash2, Mic, Eye, PowerOff } from "lucide-react";
import { SlotName, SlotStatus } from "./ModelBar";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE ?? 'http://localhost:8000';
const WHISPER_MODELS = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"];

const SLOT_DISPLAY = [
  { slot: "llm"   as SlotName, label: "LLM",           category: "LLM",             Icon: Cpu      },
  { slot: "embed" as SlotName, label: "Embedding",     category: "Embedding model", Icon: Database },
  { slot: "ocr"   as SlotName, label: "OCR",           category: "OCR model",       Icon: Eye      },
  { slot: "asr"   as SlotName, label: "ASR (Whisper)", category: "__asr__",         Icon: Mic      },
];

interface ModelSelectionModalProps {
  isOpen: boolean;
  onClose: () => void;
  slots: SlotStatus;
  onMount:   (slot: SlotName, model: string) => Promise<void>;
  onUnmount: (slot: SlotName) => Promise<void>;
  busySlot: SlotName | null;
}

export default function ModelSelectionModal({
  isOpen,
  onClose,
  slots,
  onMount,
  onUnmount,
  busySlot,
}: ModelSelectionModalProps) {
  const [models, setModels] = useState<string[]>([]);
  const [categorizedModels, setCategorizedModels] = useState<Record<string, string[]>>({});
  const [popularModels, setPopularModels] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [customModel, setCustomModel] = useState('');
  const [pullProgress, setPullProgress] = useState<{ status: string; completed?: number; total?: number } | null>(null);
  const [pullingModel, setPullingModel] = useState('');
  const abortControllerRef = useRef<AbortController | null>(null);

  // ASR
  const [asrModels, setAsrModels] = useState<string[]>([]);
  const [allAsrModels, setAllAsrModels] = useState<string[]>([]);

  const cancelDownload = () => {
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;
    setError('Download cancelled.');
    setLoading(false);
    setPullProgress(null);
    setPullingModel('');
  };

  useEffect(() => {
    if (isOpen) fetchModels();
  }, [isOpen]);

  const fetchModels = async () => {
    try {
      setLoading(true);
      const [modelsRes, trendingRes, whisperRes] = await Promise.all([
        fetch(`${API_BASE_URL}/api/models`),
        fetch(`${API_BASE_URL}/api/trending-models`),
        fetch(`${API_BASE_URL}/api/whisper/models`),
      ]);
      const modelsData   = await modelsRes.json();
      const trendingData = await trendingRes.json();
      const whisperData  = await whisperRes.json();

      setModels(modelsData.models || []);
      setCategorizedModels(modelsData.categorized_models || {});
      setPopularModels(trendingData.popular || []);
      setAsrModels(whisperData.models || []);
      setAllAsrModels(whisperData.all_models || []);
    } catch {
      setError('Failed to fetch available models.');
    } finally {
      setLoading(false);
    }
  };

  const pullAndMount = async (modelName: string, targetSlot?: SlotName) => {
    if (!modelName) return;
    setPullingModel(modelName);
    setPullProgress(null);

    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      setLoading(true);
      setError(null);

      const res = await fetch(`${API_BASE_URL}/api/models/mount`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ slot: targetSlot ?? inferSlot(modelName, categorizedModels), model_name: modelName }),
        signal: controller.signal,
      });

      const contentType = res.headers.get('content-type') || '';
      if (contentType.includes('application/x-ndjson')) {
        const reader = res.body!.getReader();
        const dec = new TextDecoder();
        let buf = '';
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buf += dec.decode(value, { stream: true });
          const lines = buf.split('\n');
          buf = lines.pop() || '';
          for (const line of lines) {
            if (!line.trim()) continue;
            try {
              const d = JSON.parse(line);
              if (d.status === 'error') throw new Error(d.message);
              if (d.status === 'success') {
                setPullProgress(null);
                await onMount(targetSlot ?? inferSlot(modelName, categorizedModels), modelName);
                return;
              }
              setPullProgress(d);
            } catch (e) {
              if (e instanceof Error && !e.message.startsWith('Unexpected')) throw e;
            }
          }
        }
      } else {
        const d = await res.json();
        if (d.status === 'error') throw new Error(d.message);
      }
    } catch (err: unknown) {
      if (err instanceof Error && err.name !== 'AbortError') {
        setError(err.message || 'Failed to mount model.');
      }
    } finally {
      if (abortControllerRef.current === controller) {
        abortControllerRef.current = null;
        setLoading(false);
        setPullingModel('');
        setPullProgress(null);
        await fetchModels();
      }
    }
  };

  const deleteSpecificModel = async (modelToDelete: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!confirm(`Delete ${modelToDelete}?`)) return;
    try {
      setLoading(true);
      setError(null);
      const res = await fetch(`${API_BASE_URL}/api/models/delete`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_name: modelToDelete }),
      });
      const data = await res.json();
      if (data.status === 'error') throw new Error(data.message);
      await fetchModels();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to delete model.');
      setLoading(false);
    }
  };

  const deleteAsrModel = async (modelName: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!confirm(`Delete ${modelName} from cache?`)) return;
    try {
      setLoading(true);
      setError(null);
      const res = await fetch(`${API_BASE_URL}/api/whisper/models/delete`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_name: modelName }),
      });
      const data = await res.json();
      if (data.status === 'error') throw new Error(data.message);
      await fetchModels();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to delete ASR model.');
    } finally {
      setLoading(false);
    }
  };

  const unmountAll = async () => {
    const activeSlots = (Object.entries(slots) as [SlotName, string | null][])
      .filter(([, v]) => v !== null)
      .map(([slot]) => slot);
    for (const slot of activeSlots) {
      await onUnmount(slot);
    }
  };

  return (
    <div
      className={`flex flex-col bg-panel/40 backdrop-blur-2xl border border-panel-border rounded-2xl transition-all duration-300 ease-[cubic-bezier(0.16,1,0.3,1)] my-3 mr-3 relative z-20 ${
        isOpen
          ? "w-[300px] opacity-100 translate-x-0"
          : "w-0 opacity-0 translate-x-10 !mr-0 !border-transparent"
      } overflow-hidden shrink-0 shadow-2xl`}
    >
      {/* Fixed-width inner so content never wraps during slide animation */}
      <div className="flex flex-col h-full w-[300px] overflow-hidden">

        {/* Header */}
        <div className="flex items-center justify-between px-4 pt-4 pb-3 shrink-0">
          <h2 className="text-sm font-semibold text-foreground tracking-tight">Manage Models</h2>
          <button
            onClick={onClose}
            className="p-1.5 text-muted hover:text-foreground hover:bg-accent/80 rounded-xl transition-colors"
          >
            <X size={16} />
          </button>
        </div>

        {/* Scrollable content */}
        <div className="flex-1 overflow-y-auto px-3 pb-4 flex flex-col gap-5 no-scrollbar">

          {/* Error */}
          {error && (
            <div className="rounded-xl bg-red-500/10 px-3 py-2.5 text-xs text-red-500 border border-red-500/20">
              <span className="font-semibold">Error: </span>{error}
            </div>
          )}

          {/* Pull progress bar */}
          {pullProgress && (
            <div className="rounded-xl bg-background/50 border border-panel-border px-3 py-2.5 flex flex-col gap-2">
              <div className="flex justify-between items-center">
                <span className="text-xs font-semibold text-foreground/90 capitalize">{pullProgress.status}</span>
                <div className="flex items-center gap-2">
                  {pullProgress.total && pullProgress.completed && (
                    <span className="text-[10px] font-medium text-muted">
                      {Math.round((pullProgress.completed / pullProgress.total) * 100)}%
                    </span>
                  )}
                  <button
                    onClick={cancelDownload}
                    className="text-[10px] font-bold uppercase tracking-wider bg-red-500/10 text-red-500 hover:bg-red-500/20 px-2 py-0.5 rounded transition-colors"
                  >
                    Cancel
                  </button>
                </div>
              </div>
              {pullProgress.total && pullProgress.completed ? (
                <div className="w-full bg-accent rounded-full h-1.5 overflow-hidden">
                  <div
                    className="bg-primary h-1.5 rounded-full transition-all duration-300 ease-out"
                    style={{ width: `${Math.round((pullProgress.completed / pullProgress.total) * 100)}%` }}
                  />
                </div>
              ) : (
                <div className="w-full bg-accent rounded-full h-1.5 overflow-hidden">
                  <div className="bg-primary h-1.5 rounded-full w-1/3 animate-pulse" />
                </div>
              )}
            </div>
          )}

          {/* ---- Mounted Slots ---- */}
          <section className="flex flex-col gap-2">
            <div className="flex items-center justify-between px-1">
              <h3 className="text-[10px] font-bold text-muted uppercase tracking-wider">Slots</h3>
              {Object.values(slots).some(Boolean) && (
                <button
                  onClick={unmountAll}
                  disabled={busySlot !== null}
                  className="flex items-center gap-1 text-[10px] font-semibold text-red-500/70 hover:text-red-500 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                  title="Unmount all models"
                >
                  <PowerOff size={10} />
                  Unmount all
                </button>
              )}
            </div>
            <div className="flex flex-col gap-1.5">
              {SLOT_DISPLAY.map(({ slot, label, Icon }) => {
                const active = slots[slot];
                const isBusy = busySlot === slot;
                return (
                  <div
                    key={slot}
                    className={`flex items-center gap-3 px-3 py-2.5 rounded-xl border transition-all ${
                      active ? "border-primary/40 bg-primary/5" : "border-panel-border/60 bg-background/30"
                    }`}
                  >
                    <div className={`p-1.5 rounded-lg shrink-0 ${active ? "bg-primary/15 text-primary" : "bg-panel text-muted"}`}>
                      <Icon size={14} />
                    </div>
                    <div className="flex-1 min-w-0">
                      <span className="text-[9px] font-bold text-muted uppercase tracking-wider">{label}</span>
                      <p className="text-xs font-semibold text-foreground/90 truncate mt-0.5">
                        {active ?? "No model mounted"}
                      </p>
                    </div>
                    {isBusy ? (
                      <RefreshCcw size={12} className="text-primary animate-spin shrink-0" />
                    ) : active ? (
                      <button
                        onClick={() => onUnmount(slot)}
                        className="flex items-center gap-1 px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-500 border border-emerald-500/20 text-[9px] font-bold uppercase hover:bg-red-500/10 hover:text-red-500 hover:border-red-500/20 transition-all group shrink-0"
                        title="Click to unmount"
                      >
                        <span className="group-hover:hidden">Active</span>
                        <span className="hidden group-hover:inline">Unmount</span>
                      </button>
                    ) : (
                      <span className="text-[9px] text-muted/50 font-medium shrink-0">idle</span>
                    )}
                  </div>
                );
              })}
            </div>
          </section>

          <div className="border-t border-panel-border/40" />

          {/* ---- ASR / Whisper ---- */}
          <section className="flex flex-col gap-2">
            <h3 className="text-[10px] font-bold text-muted uppercase tracking-wider px-1">ASR Models (Whisper)</h3>
            <div className="flex flex-col gap-1.5">
              {(allAsrModels.length > 0 ? allAsrModels : WHISPER_MODELS).map((model) => {
                const isActive    = slots.asr === model;
                const isInstalled = asrModels.includes(model);
                return (
                  <div
                    key={model}
                    className={`flex items-center gap-2.5 px-3 py-2 rounded-xl border transition-all group/item ${
                      isActive
                        ? "border-primary/80 bg-accent/60"
                        : "border-panel-border bg-background/50 hover:bg-accent/40"
                    } ${busySlot === "asr" ? "opacity-70 pointer-events-none" : ""}`}
                  >
                    <div className="p-1.5 bg-blue-500/10 text-blue-500 rounded-lg shrink-0">
                      <Mic size={13} />
                    </div>
                    <div
                      className="flex-1 min-w-0 cursor-pointer"
                      onClick={() => isActive ? onUnmount("asr") : onMount("asr", model)}
                    >
                      <span className="block text-xs font-semibold text-foreground/90 truncate">{model}</span>
                      {!isActive && isInstalled && (
                        <span className="text-[9px] font-bold text-emerald-400">Installed</span>
                      )}
                    </div>
                    <div className="flex items-center gap-1.5 shrink-0">
                      {busySlot === "asr" ? (
                        <RefreshCcw size={12} className="text-primary animate-spin" />
                      ) : isActive ? (
                        <div
                          onClick={() => onUnmount("asr")}
                          className="flex items-center gap-1 px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-500 border border-emerald-500/20 text-[9px] font-bold uppercase cursor-pointer hover:bg-red-500/10 hover:text-red-500 hover:border-red-500/20 group/active"
                        >
                          <span className="group-hover/active:hidden">Active</span>
                          <span className="hidden group-hover/active:inline">unmount</span>
                        </div>
                      ) : (
                        <span
                          onClick={() => onMount("asr", model)}
                          className="text-[10px] text-muted/50 font-semibold opacity-0 group-hover/item:opacity-100 transition-opacity cursor-pointer whitespace-nowrap"
                        >
                          {isInstalled ? "Load" : "Download"}
                        </span>
                      )}
                      {(isInstalled || isActive) && (
                        <button
                          onClick={(e) => deleteAsrModel(model, e)}
                          className={`p-1 text-red-500 hover:text-red-600 hover:bg-red-500/10 rounded transition-all ${
                            isActive ? "opacity-100" : "opacity-0 group-hover/item:opacity-100"
                          }`}
                          title="Delete from cache"
                        >
                          <Trash2 size={12} />
                        </button>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </section>

          <div className="border-t border-panel-border/40" />

          {/* ---- Pull New Model ---- */}
          <section className="flex flex-col gap-2.5">
            <h3 className="text-[10px] font-bold text-muted uppercase tracking-wider px-1">Pull New Model</h3>
            <div className="flex gap-2">
              <div className="relative flex-1">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <DownloadCloud size={13} className="text-muted/60" />
                </div>
                <input
                  type="text"
                  placeholder="e.g. llama3, mistral…"
                  value={customModel}
                  onChange={(e) => setCustomModel(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && customModel.trim() && !loading)
                      pullAndMount(customModel.trim());
                  }}
                  disabled={loading}
                  className="w-full bg-background/30 border border-panel-border rounded-xl pl-8 pr-3 py-2 text-xs font-medium text-foreground focus:outline-none focus:border-muted focus:ring-2 focus:ring-accent/20 transition-all placeholder:text-muted/40 disabled:opacity-50"
                />
              </div>
              <button
                onClick={() => pullAndMount(customModel.trim())}
                disabled={loading || !customModel.trim()}
                className="flex items-center gap-1.5 rounded-xl bg-primary px-3 py-2 text-xs font-semibold text-primary-foreground hover:bg-primary/90 transition-all disabled:opacity-50 disabled:cursor-not-allowed whitespace-nowrap shrink-0"
              >
                {loading && pullingModel === customModel.trim() ? (
                  <RefreshCcw size={13} className="animate-spin" />
                ) : (
                  <DownloadCloud size={13} />
                )}
                Pull
              </button>
            </div>

            {/* Popular models */}
            {popularModels.length > 0 && (
              <div className="flex flex-col gap-2">
                <div className="flex items-center gap-1.5 px-1">
                  <TrendingUp size={11} className="text-blue-500" />
                  <span className="text-[10px] font-semibold text-muted/80">Popular</span>
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {popularModels.map((modelName) => {
                    const isDownloaded  = models.includes(modelName) || models.includes(`${modelName}:latest`);
                    const isLoadingThis = loading && pullingModel === modelName;
                    return (
                      <button
                        key={modelName}
                        onClick={() => { setCustomModel(modelName); pullAndMount(modelName); }}
                        disabled={loading}
                        className={`flex items-center gap-1 px-2.5 py-1 rounded-lg text-[11px] font-medium border transition-all ${
                          isDownloaded
                            ? "bg-emerald-500/10 text-emerald-600 border-emerald-500/20 hover:bg-emerald-500/20"
                            : "bg-background/50 text-foreground/80 border-panel-border hover:bg-accent hover:text-foreground"
                        } disabled:opacity-50 disabled:cursor-not-allowed`}
                      >
                        {isLoadingThis ? (
                          <RefreshCcw size={10} className="animate-spin" />
                        ) : isDownloaded ? (
                          <Database size={10} />
                        ) : (
                          <DownloadCloud size={10} />
                        )}
                        {modelName}
                      </button>
                    );
                  })}
                </div>
              </div>
            )}
          </section>

          <div className="border-t border-panel-border/40" />

          {/* ---- Available Local Models ---- */}
          <section className="flex flex-col gap-2.5">
            <div className="flex items-center justify-between px-1">
              <h3 className="text-[10px] font-bold text-muted uppercase tracking-wider">Local Models</h3>
              <button
                onClick={fetchModels}
                disabled={loading}
                className="flex items-center gap-1 text-[10px] font-semibold text-muted hover:text-foreground transition-colors disabled:opacity-50"
                title="Refresh"
              >
                <RefreshCcw size={11} className={loading && !models.length ? "animate-spin" : ""} />
                Refresh
              </button>
            </div>

            <div className="flex flex-col gap-1.5">
              {loading && !models.length ? (
                <div className="flex flex-col items-center justify-center py-8 gap-2 border border-dashed border-panel-border/60 rounded-xl bg-background/20">
                  <RefreshCcw size={18} className="text-muted animate-spin" />
                  <span className="text-xs font-medium text-muted">Loading…</span>
                </div>
              ) : models.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-8 gap-2 border border-dashed border-panel-border/60 rounded-xl bg-background/20">
                  <Database size={18} className="text-muted/50" />
                  <span className="text-xs font-medium text-muted">No local models found.</span>
                </div>
              ) : (
                models.map((model) => {
                  const mountedInSlot = (Object.entries(slots) as [SlotName, string | null][])
                    .find(([, v]) => v === model)?.[0] ?? null;
                  const isLoadingThis = loading && pullingModel === model;

                  return (
                    <div
                      key={model}
                      className={`flex items-center px-3 py-2.5 gap-2.5 rounded-xl border transition-all group/item ${
                        mountedInSlot
                          ? "border-primary/80 bg-accent/60"
                          : "border-panel-border bg-background/50 hover:bg-accent/40"
                      } ${loading ? "opacity-70 pointer-events-none" : ""}`}
                    >
                      <div
                        className="flex-1 min-w-0 cursor-pointer"
                        onClick={() => {
                          if (mountedInSlot) {
                            onUnmount(mountedInSlot);
                          } else {
                            onMount(inferSlot(model, categorizedModels), model);
                          }
                        }}
                      >
                        <span className="block text-xs font-semibold text-foreground/90 truncate">{model}</span>
                        {mountedInSlot && (
                          <span className="text-[9px] font-bold text-primary/70 uppercase tracking-wider">
                            {mountedInSlot}
                          </span>
                        )}
                      </div>
                      {isLoadingThis ? (
                        <RefreshCcw size={12} className="text-primary animate-spin shrink-0" />
                      ) : (
                        <div className="flex items-center gap-1.5 shrink-0">
                          {mountedInSlot ? (
                            <div
                              onClick={(e) => { e.stopPropagation(); onUnmount(mountedInSlot); }}
                              className="flex items-center gap-1 px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-500 border border-emerald-500/20 text-[9px] font-bold uppercase cursor-pointer hover:bg-red-500/10 hover:text-red-500 hover:border-red-500/20 group/active"
                            >
                              <span className="group-hover/active:hidden">Active</span>
                              <span className="hidden group-hover/active:inline">Unmount</span>
                            </div>
                          ) : (
                            <span
                              className="text-[10px] text-muted/50 font-semibold opacity-0 group-hover/item:opacity-100 transition-opacity cursor-pointer whitespace-nowrap"
                              onClick={() => onMount(inferSlot(model, categorizedModels), model)}
                            >
                              Mount
                            </span>
                          )}
                          <button
                            onClick={(e) => deleteSpecificModel(model, e)}
                            className={`p-1 text-red-500 hover:text-red-600 hover:bg-red-500/10 rounded transition-all ${
                              mountedInSlot ? "opacity-100" : "opacity-0 group-hover/item:opacity-100"
                            }`}
                            title="Delete model"
                          >
                            <Trash2 size={12} />
                          </button>
                        </div>
                      )}
                    </div>
                  );
                })
              )}
            </div>
          </section>

        </div>
      </div>
    </div>
  );
}

// ---- helpers ---------------------------------------------------------------

function inferSlot(modelName: string, categorizedModels: Record<string, string[]>): SlotName {
  const lower = modelName.toLowerCase();
  if (lower.includes("embed") || lower.includes("bge-m3"))                                       return "embed";
  if (lower.includes("vision") || lower.includes("llava") || lower.includes("ocr") || lower.includes("clip")) return "ocr";
  if (lower.includes("whisper") || lower.includes("audio"))                                      return "asr";
  if ((categorizedModels["Embedding model"] ?? []).includes(modelName))                          return "embed";
  if ((categorizedModels["OCR model"]       ?? []).includes(modelName))                          return "ocr";
  return "llm";
}
