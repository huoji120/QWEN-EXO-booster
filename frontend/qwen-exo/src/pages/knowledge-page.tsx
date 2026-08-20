import { useEffect, useMemo, useRef, useState } from "react";
import {
  BookOpen,
  CheckCircle2,
  CircleAlert,
  Database,
  FileText,
  LoaderCircle,
  Plus,
  RefreshCw,
  Search,
  Trash2,
  Upload,
} from "lucide-react";
import { toast } from "sonner";
import { EmptyState } from "@/components/empty-state";
import { PageHeader } from "@/components/page-header";
import { TagInput } from "@/components/tag-input";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import {
  compileSourceDocuments,
  deleteSource,
  deleteSourceDocuments,
  getReflectionMemoryOrganizationStatus,
  getSource,
  createDocumentCategory,
  listDocumentCategories,
  listSources,
  previewKnowledgeFiles,
  reindexSource,
  reindexTensorBank,
  saveSource,
  startReflectionMemoryOrganization,
  type ReflectionOrganizationJobStatus,
} from "@/lib/api";
import { translate as t, translateFor, useI18n } from "@/lib/i18n";
import type { DocumentCategory, SourceDocument, SourceListing } from "@/lib/types";
import {
  cn,
  formatBytes,
  formatNumber,
  formatTime,
  shortHash,
} from "@/lib/utils";

type SourceLane = "knowledge" | "policydata";
type SourceView = SourceLane | "reflection_memory";
type SortOrder =
  | "time-desc"
  | "time-asc"
  | "path-asc"
  | "path-desc"
  | "tokens-desc"
  | "tokens-asc"
  | "status";

const LANE_COPY: Record<SourceView, { label: string; description: string }> = {
  knowledge: {
    label: "知识资料",
    description: "经相关性准入后恢复的事实、SDK 与领域参考。",
  },
  policydata: {
    label: "人格文档",
    description: "唯一、始终生效的人格与执行文档。",
  },
  reflection_memory: {
    label: "反思记忆",
    description:
      "由已完成的外部工具轨迹提炼；可直接查看、编辑、删除和重新编译。",
  },
};

function storageLane(view: SourceView): SourceLane {
  return view === "policydata" ? "policydata" : "knowledge";
}

function isReflectionMemory(document: SourceDocument) {
  return (
    document.source_kind === "trajectory_reflection" ||
    document.document_group === "reflection_memory" ||
    document.tags?.includes("reflection-memory") === true
  );
}

const SYSTEM_TAG_LABELS: Record<string, string> = {
  "reflection-memory": "反思记忆",
  "outcome-success": "结果：成功",
  "outcome-failure": "结果：失败",
  "outcome-mixed": "结果：部分完成",
  "outcome-uncertain": "结果：未确定",
  knowledge: "知识资料",
  policydata: "人格文档",
  "policy-data": "人格文档",
  "trajectory-reflection": "轨迹反思",
  "local-verified": "本地已验证",
};

function tagLabel(tag: string) {
  const systemLabel = SYSTEM_TAG_LABELS[tag.toLowerCase()];
  return systemLabel ? t(systemLabel) : tag;
}

const SOURCE_FAMILY_LABELS: Record<string, string> = {
  trajectory_reflection: "反思记忆",
  boeing_fable5_agent_trajectory: "Fable 轨迹",
  uploaded_markdown: "上传 Markdown",
  uploaded_structured_text: "上传结构化文本",
  uploaded_text: "上传文本",
  curated_reference: "整理参考",
  coding_agent_execution_policy: "执行策略",
  unclassified: "未分类",
};

function sourceFamily(document: SourceDocument) {
  return (
    document.retrieval_diversity_bucket ||
    document.source_kind ||
    "unclassified"
  );
}

function sourceFamilyLabel(sourceKind: string) {
  return t(SOURCE_FAMILY_LABELS[sourceKind] || sourceKind);
}

function markdownSourceFamily(content: string) {
  const match = content.match(/^source_kind:\s*["']?([^\n"']+)["']?\s*$/im);
  return match?.[1]?.trim() || "unclassified";
}

const ORGANIZATION_STEPS = [
  "扫描记忆",
  "Q×K 检索",
  "模型审查",
  "写入与编译",
  "完成",
];

const INITIAL_ORGANIZATION_STATUS: ReflectionOrganizationJobStatus = {
  job_id: null,
  status: "idle",
  stage: "idle",
  progress: 0,
  message: "尚未开始整理",
  details: {},
  result: null,
  error: null,
};

const ORGANIZATION_STATUS_LABELS: Record<
  ReflectionOrganizationJobStatus["status"],
  string
> = {
  idle: "未运行",
  queued: "已排队",
  running: "后台运行中",
  succeeded: "已完成",
  failed: "失败",
};

function organizationStepIndex(status: ReflectionOrganizationJobStatus) {
  if (status.status === "succeeded" || status.stage === "completed") return 4;
  if (status.stage === "publishing") return 3;
  if (status.stage === "model_review") return 2;
  if (status.stage === "qk_retrieval") return 1;
  if (status.stage === "scanning" || status.stage === "queued") return 0;
  if (status.status === "failed") {
    return Math.min(3, Math.max(0, Math.floor(status.progress / 25)));
  }
  return -1;
}

function organizationFacts(status: ReflectionOrganizationJobStatus) {
  const details = status.details || {};
  const facts: string[] = [];
  if (details.pass_index)
    facts.push(
      t("第 {count} 轮", { count: formatNumber(Number(details.pass_index)) }),
    );
  if (details.document_count !== undefined) {
    facts.push(
      t("文档 {count}", {
        count: formatNumber(Number(details.document_count)),
      }),
    );
  }
  if (details.high_qk_pair_count !== undefined) {
    facts.push(
      t("高 Q×K 候选 {count}", {
        count: formatNumber(Number(details.high_qk_pair_count)),
      }),
    );
  }
  if (details.review_count !== undefined) {
    facts.push(
      t("已审查 {count} 组", {
        count: formatNumber(Number(details.review_count)),
      }),
    );
  }
  if (status.result?.merge_operation_count !== undefined) {
    facts.push(
      t("合并 {count} 次", {
        count: formatNumber(status.result.merge_operation_count),
      }),
    );
  }
  return facts;
}

function organizationMessage(status: ReflectionOrganizationJobStatus) {
  const details = status.details || {};
  const pass = Number(details.pass_index || 0);
  if (status.status === "succeeded") {
    if (status.result?.status === "merged") {
      return t("整理完成，执行 {count} 次合并", {
        count: formatNumber(status.result.merge_operation_count || 0),
      });
    }
    if (status.result?.status === "kept_distinct") {
      return t("整理完成，模型判定候选应保持分开");
    }
    return t("整理完成，没有需要合并的高 Q×K 记忆");
  }
  if (status.status === "failed")
    return t(status.message || "反思记忆整理失败");
  if (status.stage === "queued") return t("整理任务已进入后台队列");
  if (status.stage === "scanning") {
    if (details.document_count !== undefined) {
      return t("第 {pass} 轮：发现 {count} 条反思记忆", {
        pass: formatNumber(pass),
        count: formatNumber(Number(details.document_count)),
      });
    }
    return pass
      ? t("第 {pass} 轮：正在扫描反思记忆", {
          pass: formatNumber(pass),
        })
      : t("正在扫描反思记忆");
  }
  if (status.stage === "qk_retrieval") {
    if (Number(details.documents_scanned || 0) > 0) {
      return t("第 {pass} 轮：Q×K 检索 {current}/{total}", {
        pass: formatNumber(pass),
        current: formatNumber(Number(details.documents_scanned)),
        total: formatNumber(Number(details.document_count)),
      });
    }
    return t("第 {pass} 轮：正在执行严格 Q×K 候选检索", {
      pass: formatNumber(pass),
    });
  }
  if (status.stage === "model_review") {
    if (Number(details.review_count || 0) > 0) {
      return t("第 {pass} 轮：模型正在审查第 {count} 组候选", {
        pass: formatNumber(pass),
        count: formatNumber(Number(details.review_count)),
      });
    }
    return t(
      "第 {pass} 轮：发现 {count} 组高 Q×K 候选，等待模型核对因果经验与冲突",
      {
        pass: formatNumber(pass),
        count: formatNumber(Number(details.high_qk_pair_count)),
      },
    );
  }
  if (status.stage === "publishing") {
    return details.merged_document_count !== undefined
      ? t("第 {pass} 轮：合并已提交，Tensor Bank 热编译完成", {
          pass: formatNumber(pass),
        })
      : t("第 {pass} 轮：模型决定合并，正在原子写入并热编译 Tensor Bank", {
          pass: formatNumber(pass),
        });
  }
  return t(status.message || "尚未开始整理");
}

const NEW_DOCUMENT_TEMPLATE =
  "---\ntitle: \nsource_kind: unclassified\nretrieval_category: unclassified\nquality: 1.0\n---\n\n# 标题\n\n";

const MAX_UPLOAD_FILE_BYTES = 4 * 1024 * 1024;

function readFileAsBase64(file: File) {
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () =>
      reject(new Error(t("无法读取 {name}", { name: file.name })));
    reader.onload = () => {
      const result = reader.result;
      if (typeof result !== "string" || !result.includes(",")) {
        reject(new Error(t("无法编码 {name}", { name: file.name })));
        return;
      }
      resolve(result.slice(result.indexOf(",") + 1));
    };
    reader.readAsDataURL(file);
  });
}

function withRetrievalCategory(content: string, category: string) {
  const value = category.trim() || "unclassified";
  const frontMatter = /^---\s*\n([\s\S]*?)\n---\s*(?:\n|$)/;
  const match = content.match(frontMatter);
  if (!match) return `---\nretrieval_category: ${value}\n---\n\n${content}`;
  const rows = match[1]
    .split("\n")
    .filter((row) => !/^retrieval_category\s*:/i.test(row));
  rows.push(`retrieval_category: ${value}`);
  return content.replace(frontMatter, `---\n${rows.join("\n")}\n---\n\n`);
}

export function KnowledgePage() {
  const { language, locale } = useI18n();
  const [lane, setLane] = useState<SourceView>("policydata");
  const [listing, setListing] = useState<SourceListing>({ documents: [] });
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(true);
  const [compiling, setCompiling] = useState(false);
  const [batchAction, setBatchAction] = useState<"compile" | "delete" | null>(
    null,
  );
  const [organization, setOrganization] =
    useState<ReflectionOrganizationJobStatus>(INITIAL_ORGANIZATION_STATUS);
  const [organizeOpen, setOrganizeOpen] = useState(false);
  const [editorOpen, setEditorOpen] = useState(false);
  const [editorPath, setEditorPath] = useState("");
  const [suggestedPath, setSuggestedPath] = useState("");
  const [editorContent, setEditorContent] = useState("");
  const [editorTags, setEditorTags] = useState<string[]>([]);
  const [editorCategory, setEditorCategory] = useState("unclassified");
  const [editorMode, setEditorMode] = useState<"create" | "edit">("edit");
  const [categories, setCategories] = useState<DocumentCategory[]>([]);
  const [categoryOpen, setCategoryOpen] = useState(false);
  const [newCategoryId, setNewCategoryId] = useState("");
  const [newCategoryTitle, setNewCategoryTitle] = useState("");
  const [newCategoryParent, setNewCategoryParent] = useState<string | null>(null);
  const [savingCategory, setSavingCategory] = useState(false);
  const [editorLoading, setEditorLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [deleteTargets, setDeleteTargets] = useState<SourceDocument[]>([]);
  const [selectedPaths, setSelectedPaths] = useState<Set<string>>(new Set());
  const [sortOrder, setSortOrder] = useState<SortOrder>("time-desc");
  const [uploading, setUploading] = useState(false);
  const [selectedTag, setSelectedTag] = useState("");
  const categorySuggestions = useMemo(
    () => categories.map((category) => category.category_id),
    [categories],
  );
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const activeOrganizationJobRef = useRef<string | null>(null);
  const notifiedOrganizationJobsRef = useRef(new Set<string>());
  const organizing =
    organization.status === "queued" || organization.status === "running";

  const load = async (targetView = lane, targetQuery = query) => {
    setLoading(true);
    try {
      const categoryListing = await listDocumentCategories();
      setCategories(categoryListing.categories);
      const next = await listSources(
        storageLane(targetView),
        targetView === "policydata" ? "" : targetQuery,
      );
      setListing(next);
      const available = new Set(
        next.documents.map((document) => document.relative_path),
      );
      setSelectedPaths(
        (current) =>
          new Set([...current].filter((path) => available.has(path))),
      );
    } catch (error) {
      toast.error(t("知识源加载失败"), {
        description: error instanceof Error ? error.message : t("未知错误"),
      });
    } finally {
      setLoading(false);
    }
  };

  const refreshOrganizationStatus = async () => {
    try {
      const next = await getReflectionMemoryOrganizationStatus();
      setOrganization(next);
      const activeJob = activeOrganizationJobRef.current;
      if (
        !activeJob ||
        next.job_id !== activeJob ||
        (next.status !== "succeeded" && next.status !== "failed") ||
        notifiedOrganizationJobsRef.current.has(activeJob)
      ) {
        return;
      }
      notifiedOrganizationJobsRef.current.add(activeJob);
      activeOrganizationJobRef.current = null;
      if (next.status === "failed") {
        toast.error(t("反思记忆整理失败"), {
          description: next.error || next.message,
          duration: 10000,
        });
        return;
      }
      await load("reflection_memory");
      if (next.result?.status === "merged") {
        toast.success(t("相似反思记忆已在后台合并"), {
          description: t("{count} 次合并已提交并热编译。", {
            count: formatNumber(next.result.merge_operation_count || 0),
          }),
          duration: 10000,
        });
      } else if (next.result?.status === "kept_distinct") {
        toast.success(t("后台整理完成，模型判定应保持分开"));
      } else {
        toast.success(t("后台整理完成，没有需要合并的高 Q×K 记忆"));
      }
    } catch {
      // 服务重启或短暂断连时保留最近一次可见状态，下一轮继续查询。
    }
  };

  useEffect(() => {
    setSelectedTag("");
    setSelectedPaths(new Set());
  }, [lane]);

  useEffect(() => {
    if (!editorOpen || editorMode !== "create") return;
    const otherLanguage = language === "en-US" ? "zh-CN" : "en-US";
    const previousTemplate = translateFor(otherLanguage, NEW_DOCUMENT_TEMPLATE);
    setEditorContent((current) =>
      current === previousTemplate
        ? translateFor(language, NEW_DOCUMENT_TEMPLATE)
        : current,
    );
  }, [editorMode, editorOpen, language]);

  useEffect(() => {
    const timer = window.setTimeout(() => void load(lane, query), 250);
    return () => window.clearTimeout(timer);
  }, [lane, query]);

  useEffect(() => {
    void refreshOrganizationStatus();
    const timer = window.setInterval(() => {
      void refreshOrganizationStatus();
    }, 1500);
    return () => window.clearInterval(timer);
  }, []);

  const viewDocuments = useMemo(
    () =>
      listing.documents.filter((document) =>
        lane === "reflection_memory"
          ? isReflectionMemory(document)
          : lane === "knowledge"
            ? !isReflectionMemory(document)
            : true,
      ),
    [lane, listing.documents],
  );

  const allTags = useMemo(
    () =>
      lane === "policydata"
        ? []
        : Array.from(
            new Set(viewDocuments.flatMap((document) => document.tags || [])),
          ).sort((left, right) =>
            tagLabel(left).localeCompare(tagLabel(right), locale),
          ),
    [lane, locale, viewDocuments],
  );

  const documents = useMemo(() => {
    const filtered = viewDocuments.filter((document) => {
      const tags = document.tags || [];
      if (lane !== "policydata" && selectedTag && !tags.includes(selectedTag))
        return false;
      if (document.compile_status !== undefined || !query.trim()) return true;
      const needle = query.trim().toLowerCase();
      return [
        document.relative_path,
        document.title || "",
        document.source_kind || "",
        document.document_group || "",
        ...tags,
        ...tags.map(tagLabel),
      ]
        .join(" ")
        .toLowerCase()
        .includes(needle);
    });
    return [...filtered].sort((left, right) => {
      if (sortOrder === "time-desc" || sortOrder === "time-asc") {
        const result = (left.ingested_at || 0) - (right.ingested_at || 0);
        return sortOrder === "time-desc" ? -result : result;
      }
      if (sortOrder === "tokens-desc" || sortOrder === "tokens-asc") {
        const result = (left.token_count || 0) - (right.token_count || 0);
        return sortOrder === "tokens-desc" ? -result : result;
      }
      if (sortOrder === "status") {
        const result =
          Number(left.compiled === true) - Number(right.compiled === true);
        if (result) return result;
      }
      const result = left.relative_path.localeCompare(
        right.relative_path,
        locale,
      );
      return sortOrder === "path-desc" ? -result : result;
    });
  }, [lane, locale, query, selectedTag, sortOrder, viewDocuments]);

  const visiblePaths = documents.map((document) => document.relative_path);
  const allVisibleSelected =
    visiblePaths.length > 0 &&
    visiblePaths.every((path) => selectedPaths.has(path));
  const someVisibleSelected = visiblePaths.some((path) =>
    selectedPaths.has(path),
  );
  const selectedDocuments = viewDocuments.filter((document) =>
    selectedPaths.has(document.relative_path),
  );
  const documentAdminReady =
    viewDocuments.length === 0 ||
    viewDocuments.every((document) => document.compile_status !== undefined);
  const compiledCount = viewDocuments.filter(
    (document) => document.compiled === true,
  ).length;
  const uncompiledCount = viewDocuments.filter(
    (document) => document.compiled === false,
  ).length;
  const pendingRestartCount =
    viewDocuments.length - compiledCount - uncompiledCount;

  const toggleAll = () => {
    setSelectedPaths((current) => {
      const next = new Set(current);
      if (allVisibleSelected) visiblePaths.forEach((path) => next.delete(path));
      else visiblePaths.forEach((path) => next.add(path));
      return next;
    });
  };

  const toggleOne = (path: string) => {
    setSelectedPaths((current) => {
      const next = new Set(current);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });
  };

  const openNew = () => {
    setEditorMode("create");
    setEditorPath("");
    setSuggestedPath("");
    setEditorTags([]);
    setEditorContent(t(NEW_DOCUMENT_TEMPLATE));
    setEditorCategory("unclassified");
    setEditorOpen(true);
  };

  const openDocument = async (document: SourceDocument) => {
    setEditorOpen(true);
    setEditorMode("edit");
    setSuggestedPath("");
    setEditorLoading(true);
    setEditorPath(document.relative_path);
    setEditorTags(document.tags || []);
    setEditorCategory(sourceFamily(document));
    setEditorContent("");
    try {
      const payload = await getSource(
        storageLane(lane),
        document.relative_path,
      );
      setEditorContent(payload.content);
      setEditorCategory(
        payload.retrieval_category || payload.source_kind || "unclassified",
      );
      setEditorTags(payload.tags || []);
    } catch (error) {
      toast.error(t("文档读取失败"), {
        description: error instanceof Error ? error.message : t("未知错误"),
      });
      setEditorOpen(false);
    } finally {
      setEditorLoading(false);
    }
  };

  const save = async () => {
    const path = editorPath.trim().replaceAll("\\", "/");
    if (!path || !path.endsWith(".md")) {
      toast.error(t("请填写自定义 Markdown 路径，文件名必须以 .md 结尾"));
      return;
    }
    if (lane !== "policydata" && !editorTags.length) {
      toast.error(t("至少填写一个文档标签"));
      return;
    }
    if (!editorContent.trim()) {
      toast.error(t("文档内容不能为空"));
      return;
    }
    setSaving(true);
    const savedContent =
      lane === "policydata"
        ? editorContent
        : withRetrievalCategory(editorContent, editorCategory);
    try {
      await saveSource(
        storageLane(lane),
        path,
        savedContent,
        lane === "policydata" ? [] : editorTags,
      );
      await load(lane);
      setEditorOpen(false);
      toast.success(
        editorMode === "create" ? t("文档已创建") : t("文档已更新"),
        {
          description: t("当前状态：未编译"),
        },
      );
    } catch (error) {
      toast.error(t("保存失败"), {
        description: error instanceof Error ? error.message : t("未知错误"),
        duration: 8000,
      });
    } finally {
      setSaving(false);
    }
  };

  const remove = async () => {
    if (!deleteTargets.length) return;
    setBatchAction("delete");
    try {
      const paths = deleteTargets.map((document) => document.relative_path);
      if (!documentAdminReady && paths.length === 1) {
        await deleteSource(storageLane(lane), paths[0]);
      } else {
        await deleteSourceDocuments(storageLane(lane), paths);
      }
      setDeleteTargets([]);
      setSelectedPaths(new Set());
      await load(lane);
      toast.success(
        t("已删除 {count} 个文档", { count: formatNumber(paths.length) }),
        {
          description: t("Tensor Bank 需重新编译。"),
        },
      );
    } catch (error) {
      toast.error(t("删除失败"), {
        description: error instanceof Error ? error.message : t("未知错误"),
      });
    } finally {
      setBatchAction(null);
    }
  };

  const compile = async () => {
    setCompiling(true);
    try {
      await reindexSource(storageLane(lane));
      await reindexTensorBank();
      setSelectedPaths(new Set());
      await load(lane);
      toast.success(t("全部文档编译完成"));
    } catch (error) {
      toast.error(t("编译失败"), {
        description: error instanceof Error ? error.message : t("未知错误"),
        duration: 8000,
      });
    } finally {
      setCompiling(false);
    }
  };

  const compileSelected = async () => {
    if (!selectedDocuments.length) return;
    setBatchAction("compile");
    try {
      await compileSourceDocuments(
        storageLane(lane),
        selectedDocuments.map((document) => document.relative_path),
      );
      setSelectedPaths(new Set());
      await load(lane);
      toast.success(
        t("已编译 {count} 个选中文档", {
          count: formatNumber(selectedDocuments.length),
        }),
      );
    } catch (error) {
      toast.error(t("批量编译失败"), {
        description: error instanceof Error ? error.message : t("未知错误"),
        duration: 8000,
      });
    } finally {
      setBatchAction(null);
    }
  };

  const createCategory = async () => {
    if (!newCategoryId.trim() || !newCategoryTitle.trim()) {
      toast.error(t("请填写分类标识和显示名称"));
      return;
    }
    setSavingCategory(true);
    try {
      await createDocumentCategory(
        newCategoryId.trim(),
        newCategoryTitle.trim(),
        newCategoryParent,
      );
      const categoryListing = await listDocumentCategories();
      setCategories(categoryListing.categories);
      setNewCategoryId("");
      setNewCategoryTitle("");
      setNewCategoryParent(null);
      toast.success(t("分类已创建"));
    } catch (error) {
      toast.error(t("创建分类失败"), {
        description: error instanceof Error ? error.message : t("未知错误"),
      });
    } finally {
      setSavingCategory(false);
    }
  };

  const organizeMemories = async () => {
    try {
      const accepted = await startReflectionMemoryOrganization();
      activeOrganizationJobRef.current = accepted.job_id;
      setOrganization(accepted);
      setOrganizeOpen(false);
      toast.success(t("整理任务已转入后台"), {
        description: t("可以关闭弹窗或切换页面；返回反思记忆后仍可查看进度。"),
      });
    } catch (error) {
      await refreshOrganizationStatus();
      toast.error(t("后台整理任务启动失败"), {
        description: error instanceof Error ? error.message : t("未知错误"),
        duration: 10000,
      });
    }
  };

  const importFile = async (file: File) => {
    if (file.size > MAX_UPLOAD_FILE_BYTES) {
      toast.error(t("{name} 超过单文件上限", { name: file.name }), {
        description: t("单个文件不得超过 {size}。", {
          size: formatBytes(MAX_UPLOAD_FILE_BYTES),
        }),
      });
      return;
    }
    setUploading(true);
    try {
      const result = await previewKnowledgeFiles([
        {
          filename: file.name,
          content_base64: await readFileAsBase64(file),
        },
      ]);
      const draft = result.drafts[0];
      if (!draft) throw new Error(t("服务端没有返回知识草稿"));
      setLane("knowledge");
      setEditorMode("create");
      setEditorPath("");
      setSuggestedPath(draft.suggested_path);
      setEditorTags(draft.tags || []);
      setEditorCategory(draft.retrieval_category || draft.source_kind);
      setEditorContent(draft.content);
      setEditorLoading(false);
      setEditorOpen(true);
      toast.success(t("文件已解析为可编辑草稿，知识库尚未变更"), {
        description: t("检查路径、标签和正文后保存；保存后需显式编译。"),
      });
    } catch (error) {
      toast.error(t("文件解析失败，知识库未变更"), {
        description: error instanceof Error ? error.message : t("未知错误"),
        duration: 10000,
      });
    } finally {
      setUploading(false);
    }
  };

  const activeOrganizationStep = organizationStepIndex(organization);
  const organizationDetails = organizationFacts(organization);

  return (
    <div className="page-frame">
      <PageHeader
        title={t("知识库")}
        description={
          lane === "policydata"
            ? t("人格文档只保留一份；保存后重新编译。")
            : t("文档保存后进入未编译状态；可选择编译或全部编译。")
        }
        actions={
          lane === "reflection_memory" ? (
            <Button
              variant="outline"
              size="sm"
              disabled={organizing || viewDocuments.length < 2}
              onClick={() => setOrganizeOpen(true)}
            >
              {organizing ? (
                <LoaderCircle className="animate-spin" />
              ) : (
                <RefreshCw />
              )}
              {organizing ? t("整理中…") : t("整理记忆")}
            </Button>
          ) : lane === "policydata" ? (
            <div className="flex items-center gap-2">
              {viewDocuments[0] ? (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => void openDocument(viewDocuments[0])}
                >
                  <FileText />
                  {t("编辑")}
                </Button>
              ) : (
                <Button size="sm" onClick={openNew}>
                  <Plus />
                  {t("新建")}
                </Button>
              )}
              {viewDocuments[0] ? (
                <Button
                  size="sm"
                  disabled={compiling || batchAction !== null}
                  onClick={() => void compile()}
                >
                  {compiling ? (
                    <LoaderCircle className="animate-spin" />
                  ) : (
                    <RefreshCw />
                  )}
                  {t("编译")}
                </Button>
              ) : null}
            </div>
          ) : (
            <>
              <input
                ref={fileInputRef}
                type="file"
                accept=".md,.markdown,.txt,.rst,.json,.jsonl,.yaml,.yml,.csv,text/*,application/json"
                hidden
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) void importFile(file);
                  event.target.value = "";
                }}
              />
              <Button
                variant="outline"
                size="sm"
                disabled={uploading}
                onClick={() => fileInputRef.current?.click()}
              >
                {uploading ? (
                  <LoaderCircle className="animate-spin" />
                ) : (
                  <Upload />
                )}
                {uploading ? t("解析中…") : t("导入")}
              </Button>
              <Button variant="outline" size="sm" onClick={() => setCategoryOpen(true)}>
                <Database />
                {t("分类")}
              </Button>
              <Button size="sm" onClick={openNew}>
                <Plus />
                {t("新建")}
              </Button>
            </>
          )
        }
      />

      <div className="mb-4 space-y-3 border-b pb-4">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <Tabs
            value={lane}
            onValueChange={(value) => setLane(value as SourceView)}
          >
            <TabsList>
              <TabsTrigger value="policydata">{t("人格文档")}</TabsTrigger>
              <TabsTrigger value="knowledge">{t("知识资料")}</TabsTrigger>
              <TabsTrigger value="reflection_memory">
                {t("反思记忆")}
              </TabsTrigger>
            </TabsList>
          </Tabs>
          {lane !== "policydata" ? (
            <span className="text-xs text-muted-foreground">
              {t("已选 {count}", { count: formatNumber(selectedPaths.size) })}
            </span>
          ) : null}
        </div>
        {lane !== "policydata" ? (
          <div className="flex flex-col gap-2 xl:flex-row xl:items-center">
            <div className="relative min-w-0 flex-1">
              <Search className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
              <Input
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder={t("搜索路径、标题、标签或正文")}
                className="pl-9"
              />
            </div>
            <Select
              value={sortOrder}
              onValueChange={(value) => setSortOrder(value as SortOrder)}
            >
              <SelectTrigger
                className="w-full xl:w-40"
                aria-label={t("文档排序")}
              >
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="time-desc">
                  {t("入库时间：新到旧")}
                </SelectItem>
                <SelectItem value="time-asc">
                  {t("入库时间：旧到新")}
                </SelectItem>
                <SelectItem value="path-asc">{t("路径：升序")}</SelectItem>
                <SelectItem value="path-desc">{t("路径：降序")}</SelectItem>
                <SelectItem value="tokens-desc">
                  {t("令牌数：多到少")}
                </SelectItem>
                <SelectItem value="tokens-asc">
                  {t("令牌数：少到多")}
                </SelectItem>
                <SelectItem value="status">{t("未编译优先")}</SelectItem>
              </SelectContent>
            </Select>
            <div className="flex flex-wrap gap-2">
              <Button
                variant="outline"
                disabled={
                  !documentAdminReady ||
                  !selectedDocuments.length ||
                  batchAction !== null ||
                  compiling
                }
                onClick={() => void compileSelected()}
              >
                {batchAction === "compile" ? (
                  <LoaderCircle className="animate-spin" />
                ) : (
                  <RefreshCw />
                )}
                {t("编译选中")}
              </Button>
              <Button
                variant="outline"
                disabled={
                  !documentAdminReady ||
                  !selectedDocuments.length ||
                  batchAction !== null ||
                  compiling
                }
                onClick={() => setDeleteTargets(selectedDocuments)}
              >
                <Trash2 />
                {t("删除选中")}
              </Button>
              <Button
                disabled={compiling || batchAction !== null}
                onClick={() => void compile()}
              >
                {compiling ? (
                  <LoaderCircle className="animate-spin" />
                ) : (
                  <RefreshCw />
                )}
                {t("全部编译")}
              </Button>
            </div>
          </div>
        ) : null}
      </div>

      {lane === "reflection_memory" ? (
        <div className="mb-4 space-y-4 border bg-muted/20 p-4">
          <div className="flex flex-col justify-between gap-3 sm:flex-row sm:items-start">
            <div className="flex min-w-0 items-start gap-3">
              {organizing ? (
                <LoaderCircle className="mt-0.5 h-5 w-5 shrink-0 animate-spin text-primary" />
              ) : organization.status === "succeeded" ? (
                <CheckCircle2 className="mt-0.5 h-5 w-5 shrink-0 text-emerald-600" />
              ) : organization.status === "failed" ? (
                <CircleAlert className="mt-0.5 h-5 w-5 shrink-0 text-destructive" />
              ) : (
                <RefreshCw className="mt-0.5 h-5 w-5 shrink-0 text-muted-foreground" />
              )}
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-sm font-medium">
                    {t("反思记忆整理")}
                  </span>
                  <Badge
                    variant={
                      organization.status === "failed"
                        ? "destructive"
                        : "outline"
                    }
                  >
                    {t(ORGANIZATION_STATUS_LABELS[organization.status])}
                  </Badge>
                </div>
                <div className="mt-1 text-xs text-muted-foreground">
                  {organizationMessage(organization)}
                </div>
                {organizationDetails.length ? (
                  <div className="mt-1 font-mono text-[10px] text-muted-foreground">
                    {organizationDetails.join(" · ")}
                  </div>
                ) : null}
                {organization.error ? (
                  <div className="mt-1 text-xs text-destructive">
                    {organization.error}
                  </div>
                ) : null}
              </div>
            </div>
            <div className="shrink-0 text-right">
              <div className="font-mono text-sm font-semibold">
                {formatNumber(
                  Math.max(0, Math.min(100, organization.progress)),
                )}
                %
              </div>
              <div className="mt-1 text-[10px] text-muted-foreground">
                {t("服务端后台任务")}
              </div>
            </div>
          </div>
          <div className="grid grid-cols-5 gap-2">
            {ORGANIZATION_STEPS.map((step, index) => (
              <div key={step}>
                <div
                  className={cn(
                    "h-1 bg-muted",
                    index <= activeOrganizationStep &&
                      (organization.status === "failed"
                        ? "bg-destructive"
                        : "bg-primary"),
                  )}
                />
                <div className="mt-2 hidden text-[9px] text-muted-foreground sm:block">
                  {t(step)}
                </div>
              </div>
            ))}
          </div>
          {organizing ? (
            <div className="text-[11px] text-muted-foreground">
              {t(
                "任务由服务端继续执行；可以切换页面，返回后会自动恢复当前进度。",
              )}
            </div>
          ) : null}
        </div>
      ) : null}

      {lane !== "policydata" && allTags.length ? (
        <div className="mb-4 flex flex-wrap items-center gap-1.5">
          <span className="mr-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
            {t("标签筛选")}
          </span>
          <button
            type="button"
            onClick={() => setSelectedTag("")}
            className={
              selectedTag
                ? "rounded-md border px-2 py-1 text-[11px] text-muted-foreground"
                : "rounded-md bg-foreground px-2 py-1 text-[11px] text-background"
            }
          >
            {t("全部")}
          </button>
          {allTags.map((tag) => (
            <button
              type="button"
              key={tag}
              onClick={() => setSelectedTag(tag)}
              className={
                selectedTag === tag
                  ? "rounded-md bg-foreground px-2 py-1 text-[11px] text-background"
                  : "rounded-md border px-2 py-1 text-[11px] text-muted-foreground hover:bg-muted"
              }
            >
              {tagLabel(tag)}
            </button>
          ))}
        </div>
      ) : null}

      <Card className="mb-4 border-l-4 border-l-blue-600">
        <CardContent className="flex flex-col justify-between gap-3 p-4 sm:flex-row sm:items-center">
          <div>
            <div className="text-sm font-semibold">
              {t(LANE_COPY[lane].label)}
            </div>
            <p className="mt-1 text-xs leading-5 text-muted-foreground">
              {t(LANE_COPY[lane].description)}
            </p>
          </div>
          <div className="flex shrink-0 flex-wrap items-center gap-4 text-xs">
            <div>
              <span className="text-muted-foreground">{t("文档")}</span>
              <strong className="ml-2">
                {formatNumber(viewDocuments.length)}
              </strong>
            </div>
            <div>
              <span className="text-muted-foreground">{t("已编译")}</span>
              <strong className="ml-2 text-emerald-700 dark:text-emerald-400">
                {formatNumber(compiledCount)}
              </strong>
            </div>
            <div>
              <span className="text-muted-foreground">{t("未编译")}</span>
              <strong className="ml-2 text-amber-700 dark:text-amber-400">
                {formatNumber(uncompiledCount)}
              </strong>
            </div>
            {pendingRestartCount ? (
              <div>
                <span className="text-muted-foreground">{t("待重启")}</span>
                <strong className="ml-2">
                  {formatNumber(pendingRestartCount)}
                </strong>
              </div>
            ) : null}
            <div>
              <span className="text-muted-foreground">{t("摘要")}</span>
              <span className="ml-2 font-mono">
                {shortHash(listing.source_digest, 14)}
              </span>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardContent className="p-0">
          {documents.length ? (
            <Table>
              <TableHeader>
                <TableRow>
                  {lane !== "policydata" ? (
                    <TableHead className="w-11">
                      <input
                        type="checkbox"
                        aria-label={t("全选当前文档")}
                        checked={allVisibleSelected}
                        ref={(node) => {
                          if (node)
                            node.indeterminate =
                              someVisibleSelected && !allVisibleSelected;
                        }}
                        onChange={toggleAll}
                        className="h-4 w-4 accent-primary"
                      />
                    </TableHead>
                  ) : null}
                  <TableHead>{t("文档")}</TableHead>
                  <TableHead className="w-40">{t("来源族")}</TableHead>
                  <TableHead className="w-28">{t("状态")}</TableHead>
                  <TableHead className="w-28">{t("令牌数")}</TableHead>
                  <TableHead className="w-40">{t("入库时间")}</TableHead>
                  <TableHead className="w-24">{t("大小")}</TableHead>
                  <TableHead className="w-16 text-right">{t("操作")}</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {documents.map((document) => (
                  <TableRow
                    key={document.relative_path}
                    className="cursor-pointer"
                    onClick={() => void openDocument(document)}
                  >
                    {lane !== "policydata" ? (
                      <TableCell onClick={(event) => event.stopPropagation()}>
                        <input
                          type="checkbox"
                          aria-label={t("选择 {path}", {
                            path: document.relative_path,
                          })}
                          checked={selectedPaths.has(document.relative_path)}
                          onChange={() => toggleOne(document.relative_path)}
                          className="h-4 w-4 accent-primary"
                        />
                      </TableCell>
                    ) : null}
                    <TableCell>
                      <div className="flex items-center gap-3">
                        <div className="grid h-8 w-8 place-items-center rounded-md border bg-muted">
                          <FileText className="h-4 w-4" />
                        </div>
                        <div>
                          <div className="font-mono text-xs font-medium">
                            {document.relative_path}
                          </div>
                          <div className="mt-1 text-[11px] text-muted-foreground">
                            {document.title ||
                              (lane === "policydata"
                                ? t("人格与执行文档")
                                : lane === "reflection_memory"
                                  ? t("轨迹反思记忆")
                                  : t("知识参考文档"))}
                          </div>
                          {lane !== "policydata" ? (
                            <div className="mt-1.5 flex flex-wrap gap-1">
                              {(document.tags || []).length ? (
                                document.tags?.map((tag) => (
                                  <Badge
                                    key={tag}
                                    variant="secondary"
                                    className="px-1.5 py-0 text-[10px]"
                                  >
                                    {tagLabel(tag)}
                                  </Badge>
                                ))
                              ) : (
                                <span className="text-[10px] text-muted-foreground">
                                  {t("未标记旧文档")}
                                </span>
                              )}
                            </div>
                          ) : null}
                        </div>
                      </div>
                    </TableCell>
                    <TableCell>
                      <Badge variant="secondary" className="font-mono text-[10px]">
                        {sourceFamilyLabel(sourceFamily(document))}
                      </Badge>
                    </TableCell>
                    <TableCell>
                      <Badge
                        variant="outline"
                        className={
                          document.compiled === undefined
                            ? "text-muted-foreground"
                            : document.compiled
                              ? "border-emerald-300 text-emerald-700 dark:border-emerald-800 dark:text-emerald-400"
                              : "border-amber-300 text-amber-700 dark:border-amber-800 dark:text-amber-400"
                        }
                      >
                        {document.compiled === undefined
                          ? t("待重启")
                          : document.compiled
                            ? t("已编译")
                            : t("未编译")}
                      </Badge>
                    </TableCell>
                    <TableCell className="font-mono text-xs text-muted-foreground">
                      {formatNumber(document.token_count)}
                    </TableCell>
                    <TableCell className="font-mono text-xs text-muted-foreground">
                      {formatTime(document.ingested_at)}
                    </TableCell>
                    <TableCell className="font-mono text-xs text-muted-foreground">
                      {formatBytes(document.byte_count)}
                    </TableCell>
                    <TableCell className="text-right">
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={(event) => {
                          event.stopPropagation();
                          setDeleteTargets([document]);
                        }}
                      >
                        <Trash2 />
                        <span className="sr-only">{t("删除")}</span>
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          ) : (
            <EmptyState
              icon={
                loading
                  ? LoaderCircle
                  : lane === "policydata"
                    ? Database
                    : BookOpen
              }
              title={
                loading
                  ? t("正在读取知识源")
                  : query
                    ? t("没有匹配文档")
                    : t("{lane} 为空", { lane: t(LANE_COPY[lane].label) })
              }
              description={
                loading
                  ? t("正在同步服务端 Markdown 索引。")
                  : query
                    ? t("清除搜索条件或换一个关键词。")
                    : lane === "reflection_memory"
                      ? t("完成的外部工具轨迹会在空闲后自动生成反思记忆。")
                      : t(
                          "创建或导入 Markdown 文档，然后编译到 Native Tensor Bank。",
                        )
              }
              actionLabel={
                !loading && !query && lane !== "reflection_memory"
                  ? t("新建文档")
                  : undefined
              }
              onAction={
                !loading && !query && lane !== "reflection_memory"
                  ? openNew
                  : undefined
              }
            />
          )}
        </CardContent>
      </Card>

      <Dialog open={categoryOpen} onOpenChange={setCategoryOpen}>
        <DialogContent className="max-w-3xl">
          <DialogHeader>
            <DialogTitle>{t("文档分类")}</DialogTitle>
            <DialogDescription>
              {t("分类保存在服务端数据库；分类标识稳定用于检索，显示名称可按需要调整。")}
            </DialogDescription>
          </DialogHeader>
          <div className="max-h-64 overflow-y-auto rounded-md border">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>{t("分类")}</TableHead>
                  <TableHead>{t("父级")}</TableHead>
                  <TableHead className="text-right">{t("文档数")}</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {categories.map((category) => (
                  <TableRow key={category.category_id}>
                    <TableCell>
                      <div className="font-medium">{category.title}</div>
                      <div className="font-mono text-[10px] text-muted-foreground">
                        {category.category_id}
                      </div>
                    </TableCell>
                    <TableCell className="font-mono text-xs text-muted-foreground">
                      {category.parent_id || "—"}
                    </TableCell>
                    <TableCell className="text-right font-mono text-xs">
                      {formatNumber(category.document_count)}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
          <div className="grid gap-3 sm:grid-cols-3">
            <div className="grid gap-1.5">
              <Label htmlFor="category-id">{t("分类标识")}</Label>
              <Input
                id="category-id"
                value={newCategoryId}
                onChange={(event) => setNewCategoryId(event.target.value)}
                placeholder="fastapi-routing"
              />
            </div>
            <div className="grid gap-1.5">
              <Label htmlFor="category-title">{t("显示名称")}</Label>
              <Input
                id="category-title"
                value={newCategoryTitle}
                onChange={(event) => setNewCategoryTitle(event.target.value)}
                placeholder={t("FastAPI 路由")}
              />
            </div>
            <div className="grid gap-1.5">
              <Label>{t("父级分类")}</Label>
              <Select
                value={newCategoryParent || "root"}
                onValueChange={(value) => setNewCategoryParent(value === "root" ? null : value)}
              >
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="root">{t("顶级")}</SelectItem>
                  {categories.map((category) => (
                    <SelectItem key={category.category_id} value={category.category_id}>
                      {category.title}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setCategoryOpen(false)}>
              {t("关闭")}
            </Button>
            <Button disabled={savingCategory} onClick={() => void createCategory()}>
              {savingCategory ? <LoaderCircle className="animate-spin" /> : <Plus />}
              {t("新建分类")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog
        open={editorOpen}
        onOpenChange={(open) => {
          if (!saving) setEditorOpen(open);
        }}
      >
        <DialogContent className="max-h-[92vh] max-w-5xl overflow-y-auto">
          <DialogHeader>
            <DialogTitle>
              {lane === "policydata"
                ? editorMode === "create"
                  ? t("新建人格文档")
                  : t("编辑人格文档")
                : editorMode === "create"
                  ? t("检查并创建知识文档")
                  : lane === "reflection_memory"
                    ? t("编辑反思记忆")
                    : t("编辑知识文档与标签")}
            </DialogTitle>
            <DialogDescription>
              {lane === "policydata"
                ? t("系统只保存这一份人格文档；保存后需重新编译。")
                : editorMode === "create"
                  ? t(
                      "上传文件只解析为草稿；必须填写自定义路径和标签，确认后才会写入知识库。",
                    )
                  : t("保存会刷新 {lane} 索引并重建对应的模型原生状态。", {
                      lane: t(LANE_COPY[lane].label),
                    })}
            </DialogDescription>
          </DialogHeader>
          <div
            className={cn(
              "grid gap-4",
              lane !== "policydata" && "sm:grid-cols-2",
            )}
          >
            <div className="grid gap-2">
              <Label htmlFor="source-path">{t("文档路径")}</Label>
              <Input
                id="source-path"
                value={editorPath}
                onChange={(event) => setEditorPath(event.target.value)}
                placeholder={
                  suggestedPath
                    ? t("例如 {path}", { path: suggestedPath })
                    : "topic/reference.md"
                }
                disabled={editorLoading || saving || editorMode === "edit"}
              />
              <p className="text-[10px] leading-4 text-muted-foreground">
                {t(
                  "上传文件名只作为建议，不会直接入库；已有文档路径不可修改。",
                )}
              </p>
            </div>
            {lane !== "policydata" ? (
              <div className="grid gap-2">
                <Label>{t("文档标签（至少一个）")}</Label>
                <TagInput
                  value={editorTags}
                  onChange={setEditorTags}
                  disabled={editorLoading || saving}
                  placeholder={t("例如 API 设计、构建流程、前端规范")}
                />
              </div>
            ) : null}
          </div>
          {lane !== "policydata" ? (
            <div className="grid gap-2">
              <Label htmlFor="retrieval-category">{t("检索分类")}</Label>
              <Input
                id="retrieval-category"
                list="retrieval-category-options"
                value={editorCategory}
                onChange={(event) => setEditorCategory(event.target.value)}
                placeholder={t("输入已有分类或直接创建新分类")}
                disabled={editorLoading || saving}
              />
              <datalist id="retrieval-category-options">
                {categorySuggestions.map((category) => (
                  <option key={category} value={category} />
                ))}
              </datalist>
              <p className="text-[10px] leading-4 text-muted-foreground">
                {t("上传时自动建议；可输入任意新分类。长文切片会继承同一分类。")}
              </p>
            </div>
          ) : null}
          <div className="grid gap-2">
            <div className="flex items-center justify-between">
              <Label htmlFor="source-content">{t("Markdown 内容")}</Label>
              <span className="text-[11px] text-muted-foreground">
                {t("{count} 字符", {
                  count: formatNumber(editorContent.length),
                })}
              </span>
            </div>
            {editorLoading ? (
              <div className="grid h-96 place-items-center border">
                <LoaderCircle className="animate-spin text-muted-foreground" />
              </div>
            ) : (
              <Textarea
                id="source-content"
                value={editorContent}
                onChange={(event) => setEditorContent(event.target.value)}
                className="source-editor min-h-[48vh] resize-y"
                spellCheck={false}
                disabled={saving}
              />
            )}
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setEditorOpen(false)}
              disabled={saving}
            >
              {t("取消")}
            </Button>
            <Button
              onClick={() => void save()}
              disabled={saving || editorLoading}
            >
              {saving ? <LoaderCircle className="animate-spin" /> : null}
              {editorMode === "create"
                ? lane === "policydata"
                  ? t("创建人格文档")
                  : t("创建文档")
                : lane === "policydata"
                  ? t("保存人格文档")
                  : t("保存内容与标签")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog
        open={organizeOpen}
        onOpenChange={(open) => {
          if (!organizing) setOrganizeOpen(open);
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{t("整理反思记忆")}</DialogTitle>
            <DialogDescription>
              {t(
                "系统先以严格 Q×K 检索高分候选，再由模型核对因果经验与冲突。只有表达同一经验的候选才会原子合并；仅主题相似的记忆会保持分开。启动后任务在服务端后台运行，可以关闭弹窗或切换页面。",
              )}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setOrganizeOpen(false)}
              disabled={organizing}
            >
              {t("取消")}
            </Button>
            <Button
              onClick={() => void organizeMemories()}
              disabled={organizing}
            >
              {organizing ? <LoaderCircle className="animate-spin" /> : null}
              {organizing ? t("整理中…") : t("开始整理")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog
        open={deleteTargets.length > 0}
        onOpenChange={(open) => {
          if (!open && batchAction !== "delete") setDeleteTargets([]);
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>
              {lane === "reflection_memory"
                ? t("删除反思记忆")
                : t("删除知识源")}
            </DialogTitle>
            <DialogDescription>
              {deleteTargets.length === 1 ? (
                <>
                  {t("将删除")}{" "}
                  <span className="font-mono text-foreground">
                    {deleteTargets[0]?.relative_path}
                  </span>
                  {t("。")}
                </>
              ) : (
                t("将删除选中的 {count} 个文档。", {
                  count: formatNumber(deleteTargets.length),
                })
              )}
              {t("删除后需重新编译 Tensor Bank。此操作无法撤销。")}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setDeleteTargets([])}
              disabled={batchAction === "delete"}
            >
              {t("取消")}
            </Button>
            <Button
              variant="destructive"
              onClick={() => void remove()}
              disabled={batchAction === "delete"}
            >
              {batchAction === "delete" ? (
                <LoaderCircle className="animate-spin" />
              ) : (
                <Trash2 />
              )}
              {t("确认删除")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
