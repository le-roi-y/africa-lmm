"use client";

import { useEffect, useRef, useState } from "react";
import {
  CheckCircle2,
  FileText,
  Loader2,
  MessageSquare,
  RefreshCw,
  Upload,
} from "lucide-react";
import Link from "next/link";

import {
  getDocuments,
  uploadDocument,
  type DocumentInfo,
} from "@/lib/api";

export default function DocumentsPage() {
  const inputRef = useRef<HTMLInputElement>(null);

  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  async function loadDocuments() {
    try {
      setError("");

      const result = await getDocuments();
      setDocuments(result.documents);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Impossible de charger les documents.",
      );
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadDocuments();
  }, []);

  async function handleUpload(
    event: React.ChangeEvent<HTMLInputElement>,
  ) {
    const file = event.target.files?.[0];

    if (!file) {
      return;
    }

    if (file.type !== "application/pdf") {
      setError("Seuls les documents PDF sont actuellement supportés.");
      return;
    }

    setUploading(true);
    setMessage("");
    setError("");

    try {
      const result = await uploadDocument(file);

      setMessage(
        `${result.filename} a été indexé avec succès — ${result.pages} page(s).`,
      );

      await loadDocuments();
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Impossible d'importer le document.",
      );
    } finally {
      setUploading(false);

      if (inputRef.current) {
        inputRef.current.value = "";
      }
    }
  }

  return (
    <div className="min-h-full bg-slate-50">
      <div className="mx-auto max-w-7xl px-4 py-10 sm:px-6 lg:px-8">
        {/* Header */}
        <section className="flex flex-col justify-between gap-6 md:flex-row md:items-end">
          <div>
            <p className="text-sm font-medium text-slate-500">
              Base documentaire
            </p>

            <h1 className="mt-2 text-3xl font-bold tracking-tight text-slate-950">
              Documents
            </h1>

            <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-600">
              Importez vos documents et laissez AFRICA-LMM extraire,
              structurer, indexer et rendre leur contenu interrogeable.
            </p>
          </div>

          <Link
            href="/chat"
            className="inline-flex items-center justify-center gap-2 rounded-xl bg-slate-950 px-5 py-3 text-sm font-medium text-white transition hover:bg-slate-800"
          >
            <MessageSquare size={17} />
            Ouvrir le Chat
          </Link>
        </section>

        {/* Upload */}
        <section className="mt-8 rounded-3xl border border-slate-200 bg-white p-8 shadow-sm">
          <div className="flex flex-col items-center text-center">
            <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-slate-100">
              <Upload size={24} className="text-slate-700" />
            </div>

            <h2 className="mt-5 text-xl font-semibold text-slate-950">
              Ajouter un document
            </h2>

            <p className="mx-auto mt-2 max-w-xl text-sm leading-6 text-slate-500">
              Importez un PDF. AFRICA-LMM extrait le texte, détecte les
              tableaux et images, applique l&apos;OCR si nécessaire et
              indexe le contenu dans Qdrant.
            </p>

            <input
              ref={inputRef}
              type="file"
              accept="application/pdf"
              onChange={handleUpload}
              className="hidden"
            />

            <button
              type="button"
              onClick={() => inputRef.current?.click()}
              disabled={uploading}
              className="mt-6 inline-flex items-center gap-2 rounded-xl bg-slate-950 px-6 py-3 text-sm font-medium text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {uploading ? (
                <Loader2 size={18} className="animate-spin" />
              ) : (
                <Upload size={18} />
              )}

              {uploading ? "Traitement en cours..." : "Choisir un PDF"}
            </button>

            {message && (
              <div className="mx-auto mt-5 flex max-w-xl items-center gap-2 rounded-xl border border-emerald-200 bg-emerald-50 p-4 text-left text-sm text-emerald-700">
                <CheckCircle2 size={18} />
                {message}
              </div>
            )}

            {error && (
              <div className="mx-auto mt-5 max-w-xl rounded-xl border border-red-200 bg-red-50 p-4 text-left text-sm text-red-700">
                {error}
              </div>
            )}
          </div>
        </section>

        {/* Documents */}
        <section className="mt-10">
          <div className="mb-5 flex items-center justify-between">
            <div>
              <h2 className="text-xl font-semibold text-slate-950">
                Base documentaire
              </h2>

              <p className="mt-1 text-sm text-slate-500">
                {documents.length} document
                {documents.length !== 1 ? "s" : ""} disponible
                {documents.length !== 1 ? "s" : ""}
              </p>
            </div>

            <button
              type="button"
              onClick={loadDocuments}
              disabled={loading}
              className="rounded-xl border border-slate-200 bg-white p-2.5 text-slate-500 shadow-sm transition hover:bg-slate-50 hover:text-slate-900 disabled:opacity-50"
              title="Actualiser"
            >
              <RefreshCw
                size={17}
                className={loading ? "animate-spin" : ""}
              />
            </button>
          </div>

          {loading ? (
            <div className="flex items-center justify-center rounded-2xl border border-slate-200 bg-white p-16">
              <Loader2
                size={24}
                className="animate-spin text-slate-400"
              />
            </div>
          ) : documents.length === 0 ? (
            <div className="rounded-2xl border border-dashed border-slate-300 bg-white p-16 text-center">
              <FileText
                size={32}
                className="mx-auto text-slate-300"
              />

              <p className="mt-4 font-medium text-slate-900">
                Aucun document
              </p>

              <p className="mt-1 text-sm text-slate-500">
                Importez votre premier PDF pour construire votre base
                documentaire.
              </p>
            </div>
          ) : (
            <div className="grid gap-4">
              {documents.map((document) => (
                <DocumentCard
                  key={document.document_id}
                  document={document}
                />
              ))}
            </div>
          )}
        </section>

        {/* Pipeline */}
        <section className="mt-10 pb-10">
          <h2 className="text-xl font-semibold text-slate-950">
            Pipeline documentaire
          </h2>

          <div className="mt-5 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <PipelineCard
              title="Extraction"
              description="Texte natif des PDF"
            />

            <PipelineCard
              title="OCR"
              description="Pages numérisées"
            />

            <PipelineCard
              title="Structure"
              description="Tableaux et images"
            />

            <PipelineCard
              title="Indexation"
              description="Embeddings + Qdrant"
            />
          </div>
        </section>
      </div>
    </div>
  );
}

function DocumentCard({
  document,
}: {
  document: DocumentInfo;
}) {
  const size =
    document.size_bytes < 1024 * 1024
      ? `${(document.size_bytes / 1024).toFixed(1)} KB`
      : `${(document.size_bytes / (1024 * 1024)).toFixed(1)} MB`;

  return (
    <div className="flex flex-col gap-4 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:flex-row sm:items-center sm:justify-between">
      <div className="flex min-w-0 items-center gap-4">
        <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-slate-100">
          <FileText size={20} className="text-slate-600" />
        </div>

        <div className="min-w-0">
          <p className="truncate font-medium text-slate-900">
            {document.filename}
          </p>

          <p className="mt-1 text-xs text-slate-500">
            {document.pages} page
            {document.pages !== 1 ? "s" : ""} · {size}
          </p>
        </div>
      </div>

      <div className="flex shrink-0 items-center gap-2 self-start rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1.5 text-xs font-medium text-emerald-700 sm:self-auto">
        <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
        {document.indexed ? "Indexé" : "En attente"}
      </div>
    </div>
  );
}

function PipelineCard({
  title,
  description,
}: {
  title: string;
  description: string;
}) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="mb-4 h-2 w-2 rounded-full bg-slate-900" />

      <h3 className="font-medium text-slate-900">{title}</h3>

      <p className="mt-1 text-sm text-slate-500">{description}</p>
    </div>
  );
}
