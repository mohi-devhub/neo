"use client";

import { X, Server } from "lucide-react";
import Image from "next/image";
import OpenAILogo from "../../public/OpenAI-white-monoblossom.png"
import AnthropicClaudeLogo from "../../public/icons8-claude-ai-96.png"
import GoogleCloudLogo from "../../public/icons8-google-cloud-96.png"
import SupabaseLogo from "../../public/supabase-logo-icon.png"

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function SettingsModal({ isOpen, onClose }: SettingsModalProps) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/40 backdrop-blur-md animate-in fade-in duration-200">
      <div 
        className="bg-panel/80 backdrop-blur-2xl border border-panel-border/60 rounded-3xl shadow-[0_0_80px_rgba(0,0,0,0.4)] w-full max-w-2xl max-h-[85vh] overflow-hidden flex flex-col ring-1 ring-white/5 animate-in slide-in-from-bottom-8 duration-300 ease-out"
      >
        <div className="flex items-center justify-between px-6 py-5 border-b border-panel-border/40">
          <h2 className="text-xl font-semibold text-foreground tracking-tight">Configure</h2>
          <button 
            onClick={onClose}
            className="p-2 text-muted hover:text-foreground hover:bg-accent/80 rounded-xl transition-colors"
          >
            <X size={20} />
          </button>
        </div>
        
        <div className="flex-1 overflow-y-auto px-6 py-8 flex flex-col gap-8 no-scrollbar">
          <section className="flex flex-col gap-6">
            <div>
              <h3 className="text-lg font-semibold text-foreground mb-1">Infrastructure Providers</h3>
              <p className="text-sm font-medium text-muted/80">Select where your services are hosted.</p>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 bg-background/20 p-5 rounded-2xl border border-panel-border/40">
              
              <div className="flex flex-col gap-3">
                <h4 className="text-sm font-semibold text-foreground/90">LLM Inference</h4>
                <div className="grid grid-cols-2 gap-2">
                  <label className="cursor-pointer group">
                    <input type="radio" name="llmProvider" className="peer sr-only" defaultChecked />
                    <div className="flex items-center p-3 gap-2.5 rounded-xl border border-panel-border bg-background/50 peer-checked:border-primary/80 peer-checked:bg-accent/60 group-hover:bg-accent/40 transition-all shadow-sm">
                      <Server className="w-4 h-4 text-foreground/80 peer-checked:text-foreground shrink-0" />
                      <span className="text-xs font-semibold text-foreground/90 truncate">Self-Hosted</span>
                    </div>
                  </label>
                  <label className="cursor-pointer group">
                    <input type="radio" name="llmProvider" className="peer sr-only" />
                    <div className="flex items-center p-3 gap-2.5 rounded-xl border border-panel-border bg-background/50 peer-checked:border-primary/80 peer-checked:bg-accent/60 group-hover:bg-accent/40 transition-all shadow-sm">
                      <Image src={GoogleCloudLogo} className="w-4 h-4 object-contain shrink-0" alt="GCP Logo"/>
                      <span className="text-xs font-semibold text-foreground/90 truncate">GCP</span>
                    </div>
                  </label>
                  <label className="cursor-pointer group">
                    <input type="radio" name="llmProvider" className="peer sr-only" />
                    <div className="flex items-center p-3 gap-2.5 rounded-xl border border-panel-border bg-background/50 peer-checked:border-primary/80 peer-checked:bg-accent/60 group-hover:bg-accent/40 transition-all shadow-sm">
                      <Image src={OpenAILogo} className="w-4 h-4 object-contain shrink-0" alt="OpenAI Logo"/>
                      <span className="text-xs font-semibold text-foreground/90 truncate">OpenAI</span>
                    </div>
                  </label>
                  <label className="cursor-pointer group">
                    <input type="radio" name="llmProvider" className="peer sr-only" />
                    <div className="flex items-center p-3 gap-2.5 rounded-xl border border-panel-border bg-background/50 peer-checked:border-primary/80 peer-checked:bg-accent/60 group-hover:bg-accent/40 transition-all shadow-sm">
                      <Image src={AnthropicClaudeLogo} className="w-4 h-4 object-contain shrink-0" alt="Anthropic Logo"/>
                      <span className="text-xs font-semibold text-foreground/90 truncate">Anthropic</span>
                    </div>
                  </label>
                </div>
              </div>

              {/* RAG Infra Provider */}
              <div className="flex flex-col gap-3">
                <h4 className="text-sm font-semibold text-foreground/90">RAG Engine</h4>
                <div className="grid grid-cols-2 gap-2">
                  <label className="cursor-pointer group">
                    <input type="radio" name="ragInfraProvider" className="peer sr-only" defaultChecked />
                    <div className="flex items-center p-3 gap-2.5 rounded-xl border border-panel-border bg-background/50 peer-checked:border-primary/80 peer-checked:bg-accent/60 group-hover:bg-accent/40 transition-all shadow-sm">
                      <Server className="w-4 h-4 text-foreground/80 peer-checked:text-foreground shrink-0" />
                      <span className="text-xs font-semibold text-foreground/90 truncate">Self-Hosted</span>
                    </div>
                  </label>
                  <label className="cursor-pointer group">
                    <input type="radio" name="ragInfraProvider" className="peer sr-only" />
                    <div className="flex items-center p-3 gap-2.5 rounded-xl border border-panel-border bg-background/50 peer-checked:border-primary/80 peer-checked:bg-accent/60 group-hover:bg-accent/40 transition-all shadow-sm">
                      <Image src={GoogleCloudLogo} className="w-4 h-4 object-contain shrink-0" alt="GCP Logo"/>
                      <span className="text-xs font-semibold text-foreground/90 truncate">GCP</span>
                    </div>
                  </label>
                </div>
              </div>

              <div className="flex flex-col gap-3">
                <h4 className="text-sm font-semibold text-foreground/90">Storage Bucket</h4>
                <div className="grid grid-cols-2 gap-2">
                  <label className="cursor-pointer group">
                    <input type="radio" name="storageProvider" className="peer sr-only" defaultChecked />
                    <div className="flex items-center p-3 gap-2.5 rounded-xl border border-panel-border bg-background/50 peer-checked:border-primary/80 peer-checked:bg-accent/60 group-hover:bg-accent/40 transition-all shadow-sm">
                      <Server className="w-4 h-4 text-foreground/80 peer-checked:text-foreground shrink-0" />
                      <span className="text-xs font-semibold text-foreground/90 truncate">Self-Hosted</span>
                    </div>
                  </label>
                  <label className="cursor-pointer group">
                    <input type="radio" name="storageProvider" className="peer sr-only" />
                    <div className="flex items-center p-3 gap-2.5 rounded-xl border border-panel-border bg-background/50 peer-checked:border-primary/80 peer-checked:bg-accent/60 group-hover:bg-accent/40 transition-all shadow-sm">
                      <Image src={SupabaseLogo} className="w-4 h-4 object-contain shrink-0" alt="Supabase Logo"/>
                      <span className="text-xs font-semibold text-foreground/90 truncate">Supabase</span>
                    </div>
                  </label>
                  <label className="cursor-pointer group col-span-2">
                    <input type="radio" name="storageProvider" className="peer sr-only" />
                    <div className="flex items-center justify-center p-3 gap-2.5 rounded-xl border border-panel-border bg-background/50 peer-checked:border-primary/80 peer-checked:bg-accent/60 group-hover:bg-accent/40 transition-all shadow-sm">
                      <Image src={GoogleCloudLogo} className="w-4 h-4 object-contain shrink-0" alt="GCP Logo"/>
                      <span className="text-xs font-semibold text-foreground/90 truncate">Google Cloud Platform</span>
                    </div>
                  </label>
                </div>
              </div>

              <div className="flex flex-col gap-3">
                <h4 className="text-sm font-semibold text-foreground/90">Vector Database</h4>
                <div className="grid grid-cols-2 gap-2">
                  <label className="cursor-pointer group">
                    <input type="radio" name="vectorDbProvider" className="peer sr-only" defaultChecked />
                    <div className="flex items-center p-3 gap-2.5 rounded-xl border border-panel-border bg-background/50 peer-checked:border-primary/80 peer-checked:bg-accent/60 group-hover:bg-accent/40 transition-all shadow-sm">
                      <Server className="w-4 h-4 text-foreground/80 peer-checked:text-foreground shrink-0" />
                      <span className="text-xs font-semibold text-foreground/90 truncate">Self-Hosted</span>
                    </div>
                  </label>
                  <label className="cursor-pointer group">
                    <input type="radio" name="vectorDbProvider" className="peer sr-only" />
                    <div className="flex items-center p-3 gap-2.5 rounded-xl border border-panel-border bg-background/50 peer-checked:border-primary/80 peer-checked:bg-accent/60 group-hover:bg-accent/40 transition-all shadow-sm">
                      <Image src={SupabaseLogo} className="w-4 h-4 object-contain shrink-0" alt="Supabase Logo"/>
                      <span className="text-xs font-semibold text-foreground/90 truncate">Supabase</span>
                    </div>
                  </label>
                  <label className="cursor-pointer group col-span-2">
                    <input type="radio" name="vectorDbProvider" className="peer sr-only" />
                    <div className="flex items-center justify-center p-3 gap-2.5 rounded-xl border border-panel-border bg-background/50 peer-checked:border-primary/80 peer-checked:bg-accent/60 group-hover:bg-accent/40 transition-all shadow-sm">
                      <Image src={GoogleCloudLogo} className="w-4 h-4 object-contain shrink-0" alt="GCP Logo"/>
                      <span className="text-xs font-semibold text-foreground/90 truncate">Google Cloud Platform</span>
                    </div>
                  </label>
                </div>
              </div>

            </div>
          </section>

          <hr className="border-panel-border/40" />

          {/* Configuration Settings */}
          <section className="flex flex-col md:flex-row gap-8">
            <div className="flex-1 flex flex-col gap-5">
              <div>
                <h3 className="text-base font-semibold text-foreground mb-1">Instance Configuration</h3>
                <p className="text-xs font-medium text-muted/80 leading-relaxed">For Self-Hosted RAG and LLM connection.</p>
              </div>

              <div className="flex flex-col gap-4">
                <div className="space-y-2">
                  <label className="text-[10px] font-bold text-foreground/60 tracking-wider uppercase">API Endpoint URL</label>
                  <input 
                    type="url" 
                    placeholder="https://your-ec2-instance..."
                    className="w-full bg-background/30 border border-panel-border rounded-xl px-4 py-2.5 text-sm font-medium text-foreground focus:outline-none focus:border-muted focus:ring-4 focus:ring-accent/20 transition-all placeholder:text-muted/40 shadow-inner"
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-[10px] font-bold text-foreground/60 tracking-wider uppercase">Auth Token (Optional)</label>
                  <input 
                    type="password" 
                    placeholder="sk-..."
                    className="w-full bg-background/30 border border-panel-border rounded-xl px-4 py-2.5 text-sm font-medium text-foreground focus:outline-none focus:border-muted focus:ring-4 focus:ring-accent/20 transition-all placeholder:text-muted/40 shadow-inner"
                  />
                </div>
              </div>
            </div>

            <div className="hidden md:block w-px bg-panel-border/40" />
            <hr className="md:hidden border-panel-border/40" />

            <div className="flex-1 flex flex-col gap-5">
              <div>
                <h3 className="text-base font-semibold text-foreground mb-1">Provider API Keys</h3>
                <p className="text-xs font-medium text-muted/80 leading-relaxed">Keys are stored locally in your browser.</p>
              </div>

              <div className="flex flex-col gap-4">
                <div className="space-y-2">
                  <label className="text-[10px] font-bold text-foreground/60 tracking-wider uppercase">OpenAI API Key</label>
                  <input 
                    type="password" 
                    placeholder="sk-proj-..."
                    className="w-full bg-background/30 border border-panel-border rounded-xl px-4 py-2.5 text-sm font-medium text-foreground focus:outline-none focus:border-muted focus:ring-4 focus:ring-accent/20 transition-all placeholder:text-muted/40 shadow-inner"
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-[10px] font-bold text-foreground/60 tracking-wider uppercase">Anthropic API Key</label>
                  <input 
                    type="password" 
                    placeholder="sk-ant-..."
                    className="w-full bg-background/30 border border-panel-border rounded-xl px-4 py-2.5 text-sm font-medium text-foreground focus:outline-none focus:border-muted focus:ring-4 focus:ring-accent/20 transition-all placeholder:text-muted/40 shadow-inner"
                  />
                </div>
              </div>
            </div>
          </section>
        </div>

        <div className="px-6 py-5 border-t border-panel-border/40 bg-background/10 flex justify-end gap-3 backdrop-blur-md">
          <button 
            onClick={onClose}
            className="px-5 py-2.5 text-sm font-semibold text-foreground/80 hover:bg-accent/80 rounded-xl transition-all border border-transparent hover:border-panel-border hover:shadow-sm"
          >
            Cancel
          </button>
          <button 
            onClick={onClose}
            className="px-5 py-2.5 text-sm font-semibold bg-primary text-primary-foreground hover:bg-primary/90 rounded-xl transition-all shadow-[0_0_20px_rgba(255,255,255,0.1)] hover:shadow-[0_0_30px_rgba(255,255,255,0.2)]"
          >
            Save Changes
          </button>
        </div>
      </div>
    </div>
  );
}
