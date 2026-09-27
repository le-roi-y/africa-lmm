"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  FileText,
  Menu,
  MessageSquare,
  Paperclip,
  Plus,
  Send,
  Trash2,
  X,
} from "lucide-react";

import {
  askQuestion,
  createConversation,
  deleteConversation,
  getConversation,
  getConversations,
  getDocumentUrl,
  getDocuments,
  uploadDocument,
  type Citation,
  type ConversationDetail,
  type ConversationSummary,
  type DocumentInfo,
} from "../../lib/api";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations: Citation[];
  createdAt: string;
}

interface Conversation {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  messages: Message[];
  documentIds: string[];
}

function mapConversation(
  conversation: ConversationDetail,
  documentIds: string[] = [],
): Conversation {
  return {
    id: conversation.id,
    title: conversation.title,
    createdAt: conversation.created_at,
    updatedAt: conversation.updated_at,
    documentIds,
    messages: conversation.messages.map((message) => ({
      id: message.id,
      role: message.role,
      content: message.content,
      citations: message.citations ?? [],
      createdAt: message.created_at,
    })),
  };
}

function mapSummary(
  conversation: ConversationSummary,
  documentIds: string[] = [],
): Conversation {
  return {
    id: conversation.id,
    title: conversation.title,
    createdAt: conversation.created_at,
    updatedAt: conversation.updated_at,
    documentIds,
    messages: [],
  };
}

export default function ChatPage() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<
    string | null
  >(null);
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [initializing, setInitializing] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const activeConversation = useMemo(
    () =>
      conversations.find(
        (conversation) => conversation.id === activeConversationId,
      ) ?? null,
    [conversations, activeConversationId],
  );

  const refreshDocuments = useCallback(async () => {
    try {
      const response = await getDocuments();
      setDocuments(response.documents);
    } catch {
      setError("Impossible de charger les documents.");
    }
  }, []);

  const loadConversation = useCallback(async (id: string) => {
    try {
      const detail = await getConversation(id);

      setConversations((current) =>
        current.map((conversation) =>
          conversation.id === id
            ? mapConversation(detail, conversation.documentIds)
            : conversation,
        ),
      );
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Conversation introuvable.";

      if (message.includes("Conversation not found")) {
        setConversations((current) => {
          const remaining = current.filter(
            (conversation) => conversation.id !== id,
          );

          if (remaining.length > 0) {
            setActiveConversationId((currentId) =>
              currentId === id ? remaining[0].id : currentId,
            );
          } else {
            setActiveConversationId(null);
          }

          return remaining;
        });

        return;
      }

      setError(message);
    }
  }, []);

  const initialize = useCallback(async () => {
    setInitializing(true);
    setError(null);

    try {
      const [conversationResponse] = await Promise.all([
        getConversations(),
        refreshDocuments(),
      ]);

      if (conversationResponse.conversations.length === 0) {
        const created = await createConversation();

        setConversations([mapConversation(created)]);
        setActiveConversationId(created.id);

        return;
      }

      const summaries = conversationResponse.conversations.map((conversation) =>
        mapSummary(conversation),
      );

      setConversations(summaries);

      const firstConversation = summaries[0];

      if (!firstConversation) {
        const created = await createConversation();

        setConversations([mapConversation(created)]);
        setActiveConversationId(created.id);

        return;
      }

      setActiveConversationId(firstConversation.id);

      await loadConversation(firstConversation.id);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Impossible de charger AFRICA-LMM.",
      );
    } finally {
      setInitializing(false);
    }
  }, [refreshDocuments, loadConversation]);

  useEffect(() => {
    void initialize();
  }, [initialize]);

  useEffect(() => {
    if (!activeConversationId || initializing) {
      return;
    }

    const conversation = conversations.find(
      (item) => item.id === activeConversationId,
    );

    if (conversation && conversation.messages.length === 0) {
      void loadConversation(activeConversationId);
    }
  }, [
    activeConversationId,
    conversations,
    initializing,
    loadConversation,
  ]);

  const handleNewConversation = async () => {
    setError(null);

    try {
      const conversation = await createConversation();

      setConversations((current) => [
        mapConversation(conversation),
        ...current,
      ]);
      setActiveConversationId(conversation.id);
      setSidebarOpen(false);
      textareaRef.current?.focus();
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Impossible de créer la conversation.",
      );
    }
  };

  const handleSelectConversation = async (id: string) => {
    setActiveConversationId(id);
    setSidebarOpen(false);

    const conversation = conversations.find((item) => item.id === id);

    if (!conversation || conversation.messages.length === 0) {
      try {
        await loadConversation(id);
      } catch (err) {
        setError(
          err instanceof Error
            ? err.message
            : "Impossible de charger la conversation.",
        );
      }
    }
  };

  const handleDeleteConversation = async (id: string) => {
    setError(null);

    try {
      await deleteConversation(id);

      const remaining = conversations.filter(
        (conversation) => conversation.id !== id,
      );

      if (remaining.length === 0) {
        const created = await createConversation();
        const nextConversation = mapConversation(created);

        setConversations([nextConversation]);
        setActiveConversationId(created.id);
        return;
      }

      setConversations(remaining);

      if (activeConversationId === id) {
        setActiveConversationId(remaining[0].id);
      }
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Impossible de supprimer la conversation.",
      );
    }
  };

  const toggleDocument = (documentId: string) => {
    if (!activeConversationId) {
      return;
    }

    setConversations((current) =>
      current.map((conversation) => {
        if (conversation.id !== activeConversationId) {
          return conversation;
        }

        const selected = conversation.documentIds.includes(documentId);

        return {
          ...conversation,
          documentIds: selected
            ? conversation.documentIds.filter((id) => id !== documentId)
            : [...conversation.documentIds, documentId],
        };
      }),
    );
  };

  const handleUpload = async (file: File) => {
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setError("Seuls les fichiers PDF sont acceptés.");
      return;
    }

    setUploading(true);
    setError(null);

    try {
      await uploadDocument(file);
      await refreshDocuments();
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Impossible d'importer le document.",
      );
    } finally {
      setUploading(false);
    }
  };

  const sendMessage = async () => {
    const trimmed = input.trim();

    if (!trimmed || loading || !activeConversationId) {
      return;
    }

    const conversation = conversations.find(
      (item) => item.id === activeConversationId,
    );

    if (!conversation) {
      return;
    }

    const conversationId = conversation.id;
    const documentIds = conversation.documentIds;

    setInput("");
    setError(null);
    setLoading(true);

    const temporaryUserMessage: Message = {
      id: `local-user-${Date.now()}`,
      role: "user",
      content: trimmed,
      citations: [],
      createdAt: new Date().toISOString(),
    };

    setConversations((current) =>
      current.map((item) =>
        item.id === conversationId
          ? {
              ...item,
              messages: [...item.messages, temporaryUserMessage],
            }
          : item,
      ),
    );

    try {
      const response = await askQuestion(trimmed, {
        conversationId,
        documentIds,
      });

      const assistantMessage: Message = {
        id: `local-assistant-${Date.now()}`,
        role: "assistant",
        content: response.answer,
        citations: response.citations,
        createdAt: new Date().toISOString(),
      };

      setConversations((current) =>
        current.map((item) =>
          item.id === conversationId
            ? {
                ...item,
                messages: [...item.messages, assistantMessage],
                updatedAt: new Date().toISOString(),
              }
            : item,
        ),
      );
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Une erreur est survenue pendant la génération.",
      );
    } finally {
      setLoading(false);
      textareaRef.current?.focus();
    }
  };

  const handleComposerKeyDown = (
    event: React.KeyboardEvent<HTMLTextAreaElement>,
  ) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void sendMessage();
    }
  };

  if (initializing) {
    return (
      <main className="flex min-h-screen items-center justify-center bg-white">
        <div className="text-sm text-slate-500">
          Chargement d&apos;AFRICA-LMM...
        </div>
      </main>
    );
  }

  return (
    <main className="flex min-h-screen bg-white text-slate-900">
      {sidebarOpen && (
        <button
          type="button"
          aria-label="Fermer le menu"
          className="fixed inset-0 z-30 bg-black/30 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      <aside
        className={`fixed inset-y-0 left-0 z-40 flex w-[300px] flex-col border-r border-slate-200 bg-slate-50 transition-transform lg:static lg:translate-x-0 ${
          sidebarOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <div className="flex h-16 items-center justify-between border-b border-slate-200 px-4">
          <div>
            <div className="text-lg font-semibold tracking-tight">
              AFRICA-LMM
            </div>
            <div className="text-xs text-slate-500">
              African Document Intelligence
            </div>
          </div>

          <button
            type="button"
            className="rounded-lg p-2 text-slate-500 hover:bg-slate-200 lg:hidden"
            onClick={() => setSidebarOpen(false)}
          >
            <X size={20} />
          </button>
        </div>

        <div className="p-3">
          <button
            type="button"
            onClick={() => void handleNewConversation()}
            className="flex w-full items-center justify-center gap-2 rounded-xl bg-slate-900 px-4 py-3 text-sm font-medium text-white transition hover:bg-slate-800"
          >
            <Plus size={18} />
            Nouvelle conversation
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-3">
          <div className="mb-2 px-2 text-xs font-semibold uppercase tracking-wider text-slate-400">
            Conversations
          </div>

          <div className="space-y-1">
            {conversations.map((conversation) => (
              <div
                key={conversation.id}
                className={`group flex items-center rounded-xl ${
                  conversation.id === activeConversationId
                    ? "bg-white shadow-sm ring-1 ring-slate-200"
                    : "hover:bg-slate-100"
                }`}
              >
                <button
                  type="button"
                  onClick={() => void handleSelectConversation(conversation.id)}
                  className="flex min-w-0 flex-1 items-center gap-3 px-3 py-3 text-left"
                >
                  <MessageSquare
                    size={17}
                    className="shrink-0 text-slate-400"
                  />
                  <span className="truncate text-sm">
                    {conversation.title}
                  </span>
                </button>

                <button
                  type="button"
                  aria-label="Supprimer la conversation"
                  onClick={() =>
                    void handleDeleteConversation(conversation.id)
                  }
                  className="mr-2 rounded-lg p-2 text-slate-400 opacity-0 transition hover:bg-slate-200 hover:text-red-600 group-hover:opacity-100"
                >
                  <Trash2 size={16} />
                </button>
              </div>
            ))}
          </div>
        </div>

        <div className="border-t border-slate-200 p-3">
          <div className="mb-2 flex items-center gap-2 px-2">
            <FileText size={16} className="text-slate-400" />
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
              Documents
            </span>
          </div>

          <div className="max-h-48 space-y-1 overflow-y-auto">
            {documents.length === 0 ? (
              <div className="px-2 py-2 text-xs text-slate-400">
                Aucun document indexé.
              </div>
            ) : (
              documents.map((document) => {
                const selected =
                  activeConversation?.documentIds.includes(
                    document.document_id,
                  ) ?? false;

                return (
                  <label
                    key={document.document_id}
                    className="flex cursor-pointer items-center gap-2 rounded-lg px-2 py-2 text-xs hover:bg-slate-100"
                  >
                    <input
                      type="checkbox"
                      checked={selected}
                      onChange={() =>
                        toggleDocument(document.document_id)
                      }
                      className="h-4 w-4 rounded border-slate-300"
                    />
                    <span className="min-w-0 truncate">
                      {document.filename}
                    </span>
                  </label>
                );
              })
            )}
          </div>
        </div>
      </aside>

      <section className="flex min-h-screen min-w-0 flex-1 flex-col">
        <header className="flex h-16 items-center border-b border-slate-200 px-4 sm:px-6">
          <button
            type="button"
            aria-label="Ouvrir le menu"
            className="mr-3 rounded-lg p-2 text-slate-500 hover:bg-slate-100 lg:hidden"
            onClick={() => setSidebarOpen(true)}
          >
            <Menu size={21} />
          </button>

          <div className="min-w-0">
            <h1 className="truncate text-sm font-semibold sm:text-base">
              {activeConversation?.title ?? "Nouvelle conversation"}
            </h1>
            <p className="text-xs text-slate-400">
              {activeConversation?.documentIds.length
                ? `${activeConversation.documentIds.length} document(s) sélectionné(s)`
                : "Recherche dans tous les documents indexés"}
            </p>
          </div>
        </header>

        {error && (
          <div className="mx-auto mt-4 w-full max-w-4xl px-4 sm:px-6">
            <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
              {error}
            </div>
          </div>
        )}

        <div className="flex-1 overflow-y-auto">
          <div className="mx-auto w-full max-w-4xl px-4 py-8 sm:px-6">
            {activeConversation?.messages.length === 0 ? (
              <div className="flex min-h-[60vh] flex-col items-center justify-center text-center">
                <div className="mb-5 rounded-2xl bg-slate-100 p-4">
                  <MessageSquare size={28} className="text-slate-600" />
                </div>

                <h2 className="text-2xl font-semibold tracking-tight">
                  Posez une question à AFRICA-LMM
                </h2>

                <p className="mt-2 max-w-xl text-sm leading-6 text-slate-500">
                  Analysez vos documents africains, retrouvez les informations
                  pertinentes et obtenez des réponses accompagnées de sources.
                </p>

                <div className="mt-8 text-center text-sm text-slate-400">
                  Posez votre question sur les documents sélectionnés.
                </div>
              </div>
            ) : (
              <div className="space-y-8">
                {(activeConversation?.messages ?? []).map((message) => (
                  <article
                    key={message.id}
                    className={
                      message.role === "user"
                        ? "ml-auto max-w-3xl"
                        : "max-w-4xl"
                    }
                  >
                    <div className="mb-2 text-xs font-semibold uppercase tracking-wider text-slate-400">
                      {message.role === "user" ? "Vous" : "AFRICA-LMM"}
                    </div>

                    <div
                      className={
                        message.role === "user"
                          ? "rounded-2xl bg-slate-900 px-4 py-3 text-sm leading-6 text-white"
                          : "text-sm leading-7 text-slate-800"
                      }
                    >
                      {message.role === "assistant" ? (
                        <div className="prose prose-slate max-w-none prose-sm">
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>
                            {message.content}
                          </ReactMarkdown>
                        </div>
                      ) : (
                        message.content
                      )}
                    </div>

                    {message.citations.length > 0 && (
                      <div className="mt-4 rounded-xl border border-slate-200 bg-slate-50 p-3">
                        <div className="mb-2 text-xs font-semibold uppercase tracking-wider text-slate-400">
                          Sources
                        </div>

                        <div className="space-y-1">
                          {message.citations.map((citation, index) => (
                            <a
                              key={`${citation.document_id}-${citation.page_number}-${index}`}
                              href={getDocumentUrl(
                                citation.document_id,
                                citation.page_number,
                              )}
                              target="_blank"
                              rel="noreferrer"
                              className="block truncate text-xs text-slate-600 hover:text-slate-900 hover:underline"
                            >
                              {citation.document_id}
                              {citation.page_number
                                ? ` — page ${citation.page_number}`
                                : ""}
                            </a>
                          ))}
                        </div>
                      </div>
                    )}
                  </article>
                ))}

                {loading && (
                  <div className="max-w-4xl">
                    <div className="mb-2 text-xs font-semibold uppercase tracking-wider text-slate-400">
                      AFRICA-LMM
                    </div>
                    <div className="text-sm text-slate-400">
                      Analyse des documents...
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        <div className="border-t border-slate-200 bg-white px-4 py-4 sm:px-6">
          <div className="mx-auto max-w-4xl">
            <div className="rounded-2xl border border-slate-300 bg-white shadow-sm focus-within:border-slate-400">
              <textarea
                ref={textareaRef}
                value={input}
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={handleComposerKeyDown}
                placeholder="Posez votre question..."
                rows={3}
                disabled={loading}
                className="w-full resize-none bg-transparent px-4 pt-4 text-sm outline-none placeholder:text-slate-400 disabled:opacity-50"
              />

              <div className="flex items-center justify-between px-3 pb-3">
                <div className="flex items-center gap-1">
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".pdf,application/pdf"
                    className="hidden"
                    onChange={(event) => {
                      const file = event.target.files?.[0];

                      if (file) {
                        void handleUpload(file);
                      }

                      event.target.value = "";
                    }}
                  />

                  <button
                    type="button"
                    aria-label="Importer un PDF"
                    disabled={uploading}
                    onClick={() => fileInputRef.current?.click()}
                    className="rounded-lg p-2 text-slate-400 transition hover:bg-slate-100 hover:text-slate-700 disabled:opacity-50"
                  >
                    <Paperclip size={19} />
                  </button>

                  {uploading && (
                    <span className="text-xs text-slate-400">
                      Import...
                    </span>
                  )}
                </div>

                <button
                  type="button"
                  disabled={!input.trim() || loading || !activeConversation}
                  onClick={() => void sendMessage()}
                  className="flex items-center gap-2 rounded-xl bg-slate-900 px-4 py-2.5 text-sm font-medium text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  <Send size={16} />
                  Envoyer
                </button>
              </div>
            </div>

            <p className="mt-2 text-center text-[11px] text-slate-400">
              Entrée pour envoyer · Maj+Entrée pour une nouvelle ligne
            </p>
          </div>
        </div>
      </section>
    </main>
  );
}
