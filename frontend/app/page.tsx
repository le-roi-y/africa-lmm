"use client";

import Link from "next/link";
import {
  ArrowRight,
  BrainCircuit,
  FileSearch,
  Image as ImageIcon,
  Languages,
  Menu,
  ShieldCheck,
  Sparkles,
  X,
} from "lucide-react";
import { useState } from "react";

export default function HomePage() {
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <main className="min-h-screen bg-white text-slate-950">
      {/* NAVBAR */}
      <header className="sticky top-0 z-50 border-b border-slate-200/80 bg-white/90 backdrop-blur-xl">
        <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-6">
          <Link href="/" className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-slate-950 text-white">
              <BrainCircuit size={20} />
            </div>

            <div>
              <div className="text-sm font-bold tracking-tight">
                AFRICA-LMM
              </div>
              <div className="hidden text-[10px] uppercase tracking-[0.2em] text-slate-400 sm:block">
                African Multimodal Intelligence
              </div>
            </div>
          </Link>

          <nav className="hidden items-center gap-7 md:flex">
            <a
              href="#features"
              className="text-sm text-slate-600 hover:text-slate-950"
            >
              Fonctionnalités
            </a>

            <a
              href="#technology"
              className="text-sm text-slate-600 hover:text-slate-950"
            >
              Technologie
            </a>

            <a
              href="#about"
              className="text-sm text-slate-600 hover:text-slate-950"
            >
              À propos
            </a>

            <a
              href="https://github.com/"
              target="_blank"
              rel="noreferrer"
              className="flex items-center gap-2 text-sm text-slate-600 hover:text-slate-950"
            >
              GitHub
            </a>

            <Link
              href="/chat"
              className="rounded-xl bg-slate-950 px-4 py-2.5 text-sm font-semibold text-white hover:bg-slate-800"
            >
              Commencer
            </Link>
          </nav>

          <button
            type="button"
            className="rounded-lg p-2 md:hidden"
            onClick={() => setMenuOpen(!menuOpen)}
            aria-label="Menu"
          >
            {menuOpen ? <X size={22} /> : <Menu size={22} />}
          </button>
        </div>

        {menuOpen && (
          <div className="border-t border-slate-200 bg-white px-6 py-5 md:hidden">
            <div className="flex flex-col gap-4">
              <a href="#features" onClick={() => setMenuOpen(false)}>
                Fonctionnalités
              </a>
              <a href="#technology" onClick={() => setMenuOpen(false)}>
                Technologie
              </a>
              <a href="#about" onClick={() => setMenuOpen(false)}>
                À propos
              </a>
              <Link
                href="/chat"
                onClick={() => setMenuOpen(false)}
                className="rounded-xl bg-slate-950 px-4 py-3 text-center font-semibold text-white"
              >
                Commencer
              </Link>
            </div>
          </div>
        )}
      </header>

      {/* HERO */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 -z-10 bg-[radial-gradient(circle_at_top_right,_rgba(15,23,42,0.08),_transparent_45%)]" />

        <div className="mx-auto grid max-w-7xl gap-16 px-6 py-24 lg:grid-cols-[1.1fr_0.9fr] lg:items-center lg:py-32">
          <div>
            <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-slate-200 bg-slate-50 px-3 py-1.5 text-xs font-medium text-slate-600">
              <Sparkles size={14} />
              Multimodal AI for African knowledge
            </div>

            <h1 className="max-w-4xl text-5xl font-semibold leading-[1.05] tracking-[-0.04em] sm:text-6xl lg:text-7xl">
              L'intelligence documentaire
              <span className="block text-slate-500">
                conçue pour l'Afrique.
              </span>
            </h1>

            <p className="mt-7 max-w-2xl text-lg leading-8 text-slate-600">
              AFRICA-LMM transforme vos PDF, rapports, tableaux et images en
              connaissances exploitables. Posez vos propres questions et
              obtenez des réponses contextualisées avec leurs sources.
            </p>

            <div className="mt-9 flex flex-col gap-3 sm:flex-row">
              <Link
                href="/chat"
                className="group inline-flex items-center justify-center gap-2 rounded-xl bg-slate-950 px-6 py-3.5 text-sm font-semibold text-white hover:bg-slate-800"
              >
                Commencer à explorer
                <ArrowRight
                  size={17}
                  className="transition-transform group-hover:translate-x-1"
                />
              </Link>

              <a
                href="#features"
                className="inline-flex items-center justify-center rounded-xl border border-slate-200 px-6 py-3.5 text-sm font-semibold text-slate-700 hover:bg-slate-50"
              >
                Découvrir AFRICA-LMM
              </a>
            </div>
          </div>

          {/* APERÇU PRODUIT */}
          <div className="rounded-3xl border border-slate-200 bg-slate-50 p-3 shadow-2xl shadow-slate-200/60">
            <div className="overflow-hidden rounded-2xl border border-slate-200 bg-white">
              <div className="flex items-center gap-2 border-b border-slate-200 px-5 py-4">
                <div className="h-2.5 w-2.5 rounded-full bg-slate-300" />
                <div className="h-2.5 w-2.5 rounded-full bg-slate-300" />
                <div className="h-2.5 w-2.5 rounded-full bg-slate-300" />
                <div className="ml-3 h-2.5 w-32 rounded-full bg-slate-100" />
              </div>

              <div className="grid min-h-[390px] grid-cols-[140px_1fr]">
                <div className="border-r border-slate-200 bg-slate-50 p-4">
                  <div className="mb-5 h-3 w-20 rounded bg-slate-200" />
                  <div className="space-y-3">
                    <div className="h-9 rounded-lg bg-white shadow-sm" />
                    <div className="h-9 rounded-lg bg-white" />
                    <div className="h-9 rounded-lg bg-white" />
                  </div>
                </div>

                <div className="p-6">
                  <div className="mb-8">
                    <div className="h-4 w-36 rounded bg-slate-200" />
                    <div className="mt-3 h-3 w-64 rounded bg-slate-100" />
                  </div>

                  <div className="space-y-5">
                    <div className="ml-auto max-w-[75%] rounded-2xl bg-slate-950 p-4">
                      <div className="h-3 w-40 rounded bg-slate-700" />
                      <div className="mt-2 h-3 w-28 rounded bg-slate-800" />
                    </div>

                    <div className="max-w-[85%] rounded-2xl border border-slate-200 p-4">
                      <div className="h-3 w-48 rounded bg-slate-200" />
                      <div className="mt-2 h-3 w-full rounded bg-slate-100" />
                      <div className="mt-2 h-3 w-4/5 rounded bg-slate-100" />
                      <div className="mt-4 h-2 w-28 rounded bg-slate-200" />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* FEATURES */}
      <section id="features" className="border-t border-slate-100 bg-slate-50">
        <div className="mx-auto max-w-7xl px-6 py-24">
          <p className="text-sm font-semibold uppercase tracking-[0.2em] text-slate-400">
            Une plateforme complète
          </p>

          <h2 className="mt-3 max-w-3xl text-3xl font-semibold tracking-tight sm:text-4xl">
            Comprendre les documents, pas seulement les rechercher.
          </h2>

          <p className="mt-4 max-w-2xl leading-7 text-slate-600">
            Une architecture combinant traitement documentaire, OCR, vision,
            recherche sémantique et génération augmentée par récupération.
          </p>

          <div className="mt-12 grid gap-5 md:grid-cols-2 lg:grid-cols-4">
            <Feature
              icon={<FileSearch size={21} />}
              title="Documents"
              text="PDF, rapports, textes et documents structurés."
            />

            <Feature
              icon={<ImageIcon size={21} />}
              title="Vision & OCR"
              text="Texte, images et tableaux issus de documents complexes."
            />

            <Feature
              icon={<BrainCircuit size={21} />}
              title="RAG"
              text="Recherche sémantique et réponses contextualisées."
            />

            <Feature
              icon={<Languages size={21} />}
              title="Multilingue"
              text="Une base adaptée aux usages francophones et anglophones."
            />
          </div>
        </div>
      </section>

      {/* TECHNOLOGY */}
      <section id="technology">
        <div className="mx-auto max-w-7xl px-6 py-24">
          <div className="grid gap-16 lg:grid-cols-2 lg:items-center">
            <div>
              <p className="text-sm font-semibold uppercase tracking-[0.2em] text-slate-400">
                Technologie
              </p>

              <h2 className="mt-3 text-3xl font-semibold tracking-tight sm:text-4xl">
                Une stack Machine Learning moderne.
              </h2>

              <p className="mt-5 max-w-xl leading-7 text-slate-600">
                AFRICA-LMM est construit comme une plateforme ML complète,
                depuis l'ingestion jusqu'à la génération de réponses.
              </p>

              <Link
                href="/chat"
                className="mt-8 inline-flex items-center gap-2 text-sm font-semibold"
              >
                Ouvrir l'espace IA
                <ArrowRight size={16} />
              </Link>
            </div>

            <div className="grid grid-cols-2 gap-3">
              {[
                "PyTorch",
                "Transformers",
                "FastAPI",
                "Qdrant",
                "Sentence Transformers",
                "PostgreSQL",
                "Next.js",
                "OCR",
              ].map((item) => (
                <div
                  key={item}
                  className="rounded-2xl border border-slate-200 p-5 text-sm font-medium text-slate-700"
                >
                  {item}
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section id="about" className="bg-slate-950 text-white">
        <div className="mx-auto max-w-7xl px-6 py-24">
          <ShieldCheck size={28} />

          <h2 className="mt-6 max-w-3xl text-3xl font-semibold tracking-tight sm:text-4xl">
            Interrogez vos documents avec une IA conçue pour les comprendre.
          </h2>

          <p className="mt-5 max-w-2xl text-lg leading-8 text-slate-300">
            Importez vos documents, sélectionnez vos sources et posez
            librement vos questions.
          </p>

          <Link
            href="/chat"
            className="mt-8 inline-flex items-center gap-2 rounded-xl bg-white px-6 py-3.5 text-sm font-semibold text-slate-950 hover:bg-slate-100"
          >
            Commencer
            <ArrowRight size={17} />
          </Link>
        </div>
      </section>

      {/* FOOTER */}
      <footer className="border-t border-slate-800 bg-slate-950">
        <div className="mx-auto flex max-w-7xl flex-col gap-2 px-6 py-7 text-sm text-slate-500 sm:flex-row sm:items-center sm:justify-between">
          <span>© 2026 AFRICA-LMM</span>
          <span>Multimodal AI for African knowledge</span>
        </div>
      </footer>
    </main>
  );
}

function Feature({
  icon,
  title,
  text,
}: {
  icon: React.ReactNode;
  title: string;
  text: string;
}) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-6">
      <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-slate-100 text-slate-700">
        {icon}
      </div>

      <h3 className="mt-5 font-semibold">{title}</h3>

      <p className="mt-2 text-sm leading-6 text-slate-500">{text}</p>
    </div>
  );
}
