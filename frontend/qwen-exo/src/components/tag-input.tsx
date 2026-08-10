import { useState } from "react";
import { Plus, X } from "lucide-react";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { translate as t } from "@/lib/i18n";

export function TagInput({
  value,
  onChange,
  disabled = false,
  placeholder,
}: {
  value: string[];
  onChange: (value: string[]) => void;
  disabled?: boolean;
  placeholder?: string;
}) {
  const [draft, setDraft] = useState("");

  const commit = (raw = draft) => {
    const additions = raw
      .split(",")
      .map((tag) => tag.trim())
      .filter(Boolean);
    if (!additions.length) return;
    if (additions.some((tag) => tag.length > 32)) {
      toast.error(t("单个标签不能超过 32 个字符"));
      return;
    }
    const next = [...value];
    const identities = new Set(value.map((tag) => tag.toLowerCase()));
    for (const tag of additions) {
      const identity = tag.toLowerCase();
      if (identities.has(identity)) continue;
      if (next.length >= 16) {
        toast.error(t("标签数量不能超过 16 个"));
        break;
      }
      identities.add(identity);
      next.push(tag);
    }
    onChange(next);
    setDraft("");
  };

  return (
    <div className="rounded-md border bg-background p-2 focus-within:ring-2 focus-within:ring-ring">
      {value.length ? (
        <div className="mb-2 flex flex-wrap gap-1.5">
          {value.map((tag) => (
            <Badge key={tag} variant="secondary" className="gap-1 pr-1">
              {tag}
              <button
                type="button"
                className="rounded p-0.5 hover:bg-foreground/10"
                disabled={disabled}
                onClick={() =>
                  onChange(value.filter((candidate) => candidate !== tag))
                }
                aria-label={t("删除标签 {tag}", { tag })}
              >
                <X className="h-3 w-3" />
              </button>
            </Badge>
          ))}
        </div>
      ) : null}
      <div className="flex gap-2">
        <Input
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" || event.key === ",") {
              event.preventDefault();
              commit();
            } else if (event.key === "Backspace" && !draft && value.length) {
              onChange(value.slice(0, -1));
            }
          }}
          onBlur={() => commit()}
          placeholder={placeholder || t("输入标签后按 Enter")}
          disabled={disabled}
          className="h-8 border-0 px-1 shadow-none focus-visible:ring-0"
        />
        <Button
          type="button"
          variant="ghost"
          size="icon"
          className="h-8 w-8 shrink-0"
          disabled={disabled || !draft.trim()}
          onMouseDown={(event) => event.preventDefault()}
          onClick={() => commit()}
          aria-label={t("添加标签")}
        >
          <Plus />
        </Button>
      </div>
    </div>
  );
}
