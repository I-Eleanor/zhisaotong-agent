import { Bot } from "lucide-react";
import { ThemeToggle } from "./ThemeToggle";

export function AppHeader() {
  return (
    <header className="flex items-center justify-between border-b px-6 py-3">
      <div className="flex items-center gap-3">
        <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary text-primary-foreground">
          <Bot className="h-6 w-6" />
        </div>
        <div>
          <h1 className="text-lg font-bold leading-tight">智扫通</h1>
          <p className="text-xs text-muted-foreground">扫地机器人智能客服</p>
        </div>
      </div>
      <ThemeToggle />
    </header>
  );
}
