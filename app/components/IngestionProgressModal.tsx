"use client";

import { X, CheckCircle2, AlertCircle, Loader2, Music, Image, FileText, File, Video } from "lucide-react";

export type IngestFileStatus = "pending" | "processing" | "done" | "error" | "duplicate" | "skipped";

export interface IngestFileEntry {
  id: string;
  name: string;
  category: "audio" | "image" | "text" | "pdf" | "video" | "unknown";
  status: IngestFileStatus;
  detail?: string;
  chunks?: number;
}

interface IngestionProgressModalProps {
  isOpen: boolean;
  onClose: () => void;
  files: IngestFileEntry[];
  overallProgress: number;
  isDone: boolean;
}


function CategoryIcon({ category }: { category: IngestFileEntry["category"] }) {
  const cls = "shrink-0";
  switch (category) {
    case "audio":   return <Music    size={15} className={`${cls} text-violet-400`} />;
    case "video":   return <Video    size={15} className={`${cls} text-pink-400`} />;
    case "image":   return <Image    size={15} className={`${cls} text-sky-400`} />;
    case "text":    return <FileText size={15} className={`${cls} text-emerald-400`} />;
    case "pdf":     return <FileText size={15} className={`${cls} text-orange-400`} />;
    default:        return <File     size={15} className={`${cls} text-muted`} />;
  }
}

function StatusIcon({ status }: { status: IngestFileStatus }) {
  switch (status) {
    case "done":
    case "duplicate":
      return <CheckCircle2 size={15} className="shrink-0 text-emerald-400" />;
    case "error":
      return <AlertCircle  size={15} className="shrink-0 text-red-400" />;
    case "skipped":
      return <AlertCircle  size={15} className="shrink-0 text-yellow-400" />;
    case "processing":
      return <Loader2      size={15} className="shrink-0 text-primary animate-spin" />;
    default:
      return <div className="shrink-0 w-[15px] h-[15px] rounded-full border border-panel-border/60" />;
  }
}

function statusLabel(entry: IngestFileEntry): { text: string; colour: string } {
  switch (entry.status) {
    case "pending":
      return { text: "waiting", colour: "text-muted/60" };
    case "processing":
      return { text: "processing…", colour: "text-primary/80" };
    case "done":
      return {
        text: entry.chunks != null && entry.chunks > 0
          ? `${entry.chunks} chunk${entry.chunks !== 1 ? "s" : ""} embedded`
          : "added to context",
        colour: "text-emerald-400",
      };
    case "duplicate":
      return {
        text: `already indexed (${entry.chunks ?? 0} chunk${(entry.chunks ?? 0) !== 1 ? "s" : ""})`,
        colour: "text-sky-400",
      };
    case "error":
      return { text: entry.detail ?? "error", colour: "text-red-400" };
    case "skipped":
      return { text: "unsupported type, skipped", colour: "text-yellow-400" };
  }
}

export default function IngestionProgressModal({
  isOpen,
  onClose,
  files,
  overallProgress,
  isDone,
}: IngestionProgressModalProps) {
  if (!isOpen) return null;

  const doneCount  = files.filter((f) => f.status === "done" || f.status === "duplicate").length;
  const errorCount = files.filter((f) => f.status === "error").length;
  const total      = files.length;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-background/60 backdrop-blur-sm">
      <div
        className="relative w-full max-w-lg mx-4 bg-panel border border-panel-border rounded-2xl shadow-2xl overflow-hidden animate-in zoom-in-95 fade-in duration-200"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-5 pt-5 pb-3">
          <div>
            <h2 className="text-base font-semibold text-foreground">
              {isDone ? "Indexing complete" : "Indexing files…"}
            </h2>
            <p className="text-xs text-muted mt-0.5">
              {isDone
                ? `${doneCount} indexed${errorCount > 0 ? `, ${errorCount} failed` : ""}`
                : `${doneCount} of ${total} done`}
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-muted hover:text-foreground hover:bg-foreground/8 transition-colors"
          >
            <X size={18} />
          </button>
        </div>

        <div className="px-5 pb-3">
          <div className="w-full bg-accent/60 rounded-full h-2 overflow-hidden">
            <div
              className={`h-2 rounded-full transition-all duration-500 ease-out ${
                isDone
                  ? errorCount > 0 && doneCount === 0
                    ? "bg-red-500"
                    : "bg-emerald-500"
                  : "bg-primary"
              }`}
              style={{ width: `${overallProgress}%` }}
            />
          </div>
          <div className="flex justify-between mt-1">
            <span className="text-[11px] text-muted/70">
              {isDone ? "Done" : `${Math.round(overallProgress)}%`}
            </span>
            <span className="text-[11px] text-muted/70">{total} file{total !== 1 ? "s" : ""}</span>
          </div>
        </div>

        <div className="px-3 pb-4 max-h-72 overflow-y-auto scrollbar-thin">
          <div className="flex flex-col gap-1">
            {files.map((entry) => {
              const { text: labelText, colour } = statusLabel(entry);
              return (
                <div
                  key={entry.id}
                  className={`flex items-center gap-2.5 px-3 py-2 rounded-xl transition-colors ${
                    entry.status === "processing"
                      ? "bg-primary/8 border border-primary/20"
                      : "bg-background/40 border border-transparent"
                  }`}
                >
                  <StatusIcon status={entry.status} />
                  <CategoryIcon category={entry.category} />
                  <span className="flex-1 text-sm text-foreground/90 truncate font-medium min-w-0">
                    {entry.name}
                  </span>
                  <span className={`text-[11px] shrink-0 font-medium ${colour}`}>
                    {labelText}
                  </span>
                </div>
              );
            })}
          </div>
        </div>

        {isDone && (
          <div className="px-5 pb-5 pt-1 flex justify-end">
            <button
              onClick={onClose}
              className="px-4 py-1.5 rounded-lg bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90 transition-colors"
            >
              Done
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
