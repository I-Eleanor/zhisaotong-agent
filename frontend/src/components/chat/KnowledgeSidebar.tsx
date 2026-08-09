import { useRef, useState } from "react";
import {
  CheckCircle2,
  HeartPulse,
  Loader2,
  RefreshCw,
  Upload,
  XCircle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { checkHealth, rebuildKnowledge, uploadKnowledge } from "@/lib/api";
import type { HealthInfo } from "@/lib/types";

export function KnowledgeSidebar() {
  const [files, setFiles] = useState<FileList | null>(null);
  const [busy, setBusy] = useState<"" | "upload" | "rebuild" | "health">("");
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const [health, setHealth] = useState<HealthInfo | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const handleUpload = async () => {
    if (!files || files.length === 0) return;
    setBusy("upload");
    setMsg(null);
    try {
      const r = await uploadKnowledge(Array.from(files));
      setMsg({ ok: true, text: `已保存 ${r.file_count} 个文件` });
      setFiles(null);
      if (fileRef.current) fileRef.current.value = "";
    } catch (e: any) {
      setMsg({ ok: false, text: e?.message || "上传失败" });
    } finally {
      setBusy("");
    }
  };

  const handleRebuild = async () => {
    setBusy("rebuild");
    setMsg(null);
    try {
      const r = await rebuildKnowledge();
      setMsg({ ok: true, text: `重建完成，当前 ${r.chunk_count} 个分块` });
    } catch (e: any) {
      setMsg({ ok: false, text: e?.message || "重建失败" });
    } finally {
      setBusy("");
    }
  };

  const handleHealth = async () => {
    setBusy("health");
    setMsg(null);
    try {
      const h = await checkHealth();
      setHealth(h);
    } catch (e: any) {
      setMsg({ ok: false, text: e?.message || "后端不可达" });
    } finally {
      setBusy("");
    }
  };

  return (
    <div className="flex h-full flex-col gap-4">
      <div>
        <h2 className="flex items-center gap-2 text-lg font-semibold">
          <Upload className="h-5 w-5 text-primary" /> 知识库管理
        </h2>
        <p className="mt-1 text-xs text-muted-foreground">
          上传文档并重建向量库，让客服回答更精准。
        </p>
      </div>

      <input
        ref={fileRef}
        type="file"
        accept=".txt,.pdf"
        multiple
        className="hidden"
        onChange={(e) => setFiles(e.target.files)}
      />
      <Button variant="outline" className="w-full" onClick={() => fileRef.current?.click()}>
        {files && files.length > 0 ? `已选择 ${files.length} 个文件` : "选择知识文档（txt / pdf）"}
      </Button>
      <Button
        className="w-full"
        disabled={!files || files.length === 0 || busy !== ""}
        onClick={handleUpload}
      >
        {busy === "upload" ? (
          <Loader2 className="h-4 w-4 animate-spin" />
        ) : (
          <Upload className="h-4 w-4" />
        )}
        保存到知识库
      </Button>

      <Button
        variant="secondary"
        className="w-full"
        disabled={busy !== ""}
        onClick={handleRebuild}
      >
        {busy === "rebuild" ? (
          <Loader2 className="h-4 w-4 animate-spin" />
        ) : (
          <RefreshCw className="h-4 w-4" />
        )}
        重建向量库
      </Button>

      <div className="my-1 border-t" />

      <Button variant="ghost" className="w-full" disabled={busy !== ""} onClick={handleHealth}>
        {busy === "health" ? (
          <Loader2 className="h-4 w-4 animate-spin" />
        ) : (
          <HeartPulse className="h-4 w-4" />
        )}
        检查服务健康
      </Button>

      {msg ? (
        <div
          className={`flex items-center gap-2 rounded-md px-3 py-2 text-sm ${
            msg.ok
              ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400"
              : "bg-destructive/10 text-destructive"
          }`}
        >
          {msg.ok ? <CheckCircle2 className="h-4 w-4" /> : <XCircle className="h-4 w-4" />}
          {msg.text}
        </div>
      ) : null}

      {health ? (
        <Card>
          <CardContent className="space-y-1 p-4 text-xs">
            <div className="flex justify-between">
              <span className="text-muted-foreground">状态</span>
              <span className="font-medium text-emerald-600">● {health.status}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">模型</span>
              <span className="font-medium">{health.model}</span>
            </div>
            <div className="flex justify-between gap-2">
              <span className="shrink-0 text-muted-foreground">Embedding</span>
              <span className="truncate font-medium" title={health.embedding}>
                {health.embedding}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Reranker</span>
              <span className="font-medium">{health.reranker_enabled ? "开启" : "关闭"}</span>
            </div>
          </CardContent>
        </Card>
      ) : null}
    </div>
  );
}
