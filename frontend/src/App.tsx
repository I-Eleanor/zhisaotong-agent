import { useEffect, useState } from "react";
import { MessageCircle, Stethoscope } from "lucide-react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { AppHeader } from "@/components/chat/AppHeader";
import { ChatPanel } from "@/components/chat/ChatPanel";
import { DiagnosticPanel } from "@/components/chat/DiagnosticPanel";
import { KnowledgeSidebar } from "@/components/chat/KnowledgeSidebar";
import { AdminHandoffs } from "@/pages/AdminHandoffs";

/** 轻量前端路由：基于 location.hash 决定是否进入管理端页面。 */
function useHashRoute(): string {
  const [hash, setHash] = useState(() => window.location.hash);
  useEffect(() => {
    const onChange = () => setHash(window.location.hash);
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return hash;
}

export default function App() {
  const hash = useHashRoute();
  const isAdmin = hash.startsWith("#/admin/handoffs");

  if (isAdmin) {
    return <AdminHandoffs />;
  }

  return (
    <div className="flex h-screen flex-col bg-background">
      <AppHeader />
      <div className="flex min-h-0 flex-1">
        <aside className="hidden w-72 shrink-0 border-r p-4 md:block">
          <KnowledgeSidebar />
        </aside>
        <main className="flex min-h-0 flex-1 flex-col px-4 py-4">
          <Tabs defaultValue="chat" className="flex min-h-0 flex-1 flex-col">
            <TabsList className="self-center">
              <TabsTrigger value="chat">
                <MessageCircle className="h-4 w-4" /> 客服问答
              </TabsTrigger>
              <TabsTrigger value="diagnose">
                <Stethoscope className="h-4 w-4" /> 设备诊断
              </TabsTrigger>
            </TabsList>
            <TabsContent value="chat" className="min-h-0 flex-1">
              <ChatPanel />
            </TabsContent>
            <TabsContent value="diagnose" className="min-h-0 flex-1">
              <DiagnosticPanel />
            </TabsContent>
          </Tabs>
        </main>
      </div>
    </div>
  );
}