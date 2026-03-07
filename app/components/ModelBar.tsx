"use client";
import React, { useRef, useEffect, useState } from "react";
import { Cpu, Database, Eye, Mic, ChevronDown, RefreshCw, Check, X } from "lucide-react";

export type SlotName = "llm" | "embed" | "ocr" | "asr";

export interface SlotStatus {
  llm:   string | null;
  embed: string | null;
  ocr:   string | null;
  asr:   string | null;
}

interface ModelBarProps {
  slots: SlotStatus;
  categorizedModels: Record<string, string[]>;
  asrModels: string[];
  onMount:   (slot: SlotName, model: string) => Promise<void>;
  onUnmount: (slot: SlotName) => Promise<void>;
  busySlot: SlotName | null;
}

const SLOT_META: { slot: SlotName; label: string; category: string; Icon: React.ElementType }[] = [
  { slot: "llm",   label: "LLM",    category: "LLM",             Icon: Cpu      },
  { slot: "embed", label: "Embed",  category: "Embedding model", Icon: Database },
  { slot: "ocr",   label: "OCR",    category: "OCR model",       Icon: Eye      },
  { slot: "asr",   label: "ASR",    category: "__asr__",         Icon: Mic      },
];

export default function ModelBar({
  slots,
  categorizedModels,
  asrModels,
  onMount,
  onUnmount,
  busySlot,
}: ModelBarProps) {
  const [openSlot, setOpenSlot] = useState<SlotName | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Close popover on outside click
  useEffect(() => {
    function handler(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpenSlot(null);
      }
    }
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  return (
    <div ref={containerRef} className="flex items-center gap-1.5">
      {SLOT_META.map(({ slot, label, category, Icon }) => {
        const active = slots[slot];
        const isBusy = busySlot === slot;
        const isOpen = openSlot === slot;

        // Models available for this slot.
        // If the strict category list is empty (e.g. embedding model name doesn't
        // match our keywords), fall back to ALL local Ollama models so the user
        // can still assign any model to any slot.
        const strictCandidates: string[] =
          slot === "asr"
            ? asrModels
            : categorizedModels[category] ?? [];

        const allOllamaModels: string[] = Object.values(categorizedModels).flat();

        const candidates: string[] =
          slot === "asr"
            ? asrModels
            : strictCandidates.length > 0
              ? strictCandidates
              : allOllamaModels;

        return (
          <div key={slot} className="relative">
            {/* Chip */}
            <button
              onClick={() => setOpenSlot(isOpen ? null : slot)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium border backdrop-blur-xl transition-all duration-200 ${
                active
                  ? "bg-primary/10 border-primary/30 text-foreground hover:bg-primary/20"
                  : "bg-panel/40 border-panel-border/60 text-muted hover:text-foreground hover:bg-panel/80"
              } ${isBusy ? "opacity-60 pointer-events-none" : ""}`}
            >
              {isBusy ? (
                <RefreshCw size={11} className="animate-spin shrink-0" />
              ) : (
                <Icon size={11} className={active ? "text-primary" : "text-muted"} />
              )}
              <span className="uppercase tracking-wider font-bold text-[10px]">{label}</span>
              {active ? (
                <span className="max-w-[90px] truncate text-foreground/80">{active}</span>
              ) : (
                <span className="text-muted/50">—</span>
              )}
              <ChevronDown size={10} className={`text-muted transition-transform ${isOpen ? "rotate-180" : ""}`} />
            </button>

            {/* Popover */}
            {isOpen && (
              <div className="absolute top-full mt-2 left-0 w-64 bg-panel border border-panel-border/60 rounded-2xl shadow-2xl z-50 overflow-hidden animate-in fade-in slide-in-from-top-2 duration-150">
                <div className="px-3 py-2.5 border-b border-panel-border/40">
                  <span className="text-[10px] font-bold uppercase tracking-wider text-muted">
                    {label} model
                  </span>
                </div>

                {candidates.length === 0 ? (
                  <div className="px-4 py-6 text-xs text-muted text-center">
                    No {label} models downloaded.
                    <br />
                    <span className="text-muted/60">Use Manage Models to pull one.</span>
                  </div>
                ) : (
                  <div className="max-h-52 overflow-y-auto py-1 no-scrollbar">
                    {candidates.map((m) => {
                      const isActive = active === m;
                      return (
                        <button
                          key={m}
                          onClick={() => {
                            if (isActive) {
                              onUnmount(slot);
                            } else {
                              onMount(slot, m);
                            }
                            setOpenSlot(null);
                          }}
                          className="w-full flex items-center justify-between px-4 py-2 text-xs hover:bg-accent/10 transition-colors group"
                        >
                          <span className={`truncate ${isActive ? "text-foreground font-semibold" : "text-muted group-hover:text-foreground"}`}>
                            {m}
                          </span>
                          {isActive ? (
                            <div className="flex items-center gap-1.5">
                              <span className="text-[9px] uppercase font-bold text-emerald-400 group-hover:hidden">active</span>
                              <span className="hidden group-hover:flex items-center gap-1 text-[9px] uppercase font-bold text-red-400">
                                <X size={9} /> unmount
                              </span>
                            </div>
                          ) : (
                            <Check size={11} className="text-primary opacity-0 group-hover:opacity-100 transition-opacity" />
                          )}
                        </button>
                      );
                    })}
                  </div>
                )}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
