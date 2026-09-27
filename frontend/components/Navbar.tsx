"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { FileText, MessageSquare, Sparkles } from "lucide-react";

const links = [
  {
    href: "/",
    label: "Accueil",
    icon: Sparkles,
  },
  {
    href: "/chat",
    label: "Chat",
    icon: MessageSquare,
  },
  {
    href: "/documents",
    label: "Documents",
    icon: FileText,
  },
];

export default function Navbar() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-50 border-b border-slate-200 bg-white/90 backdrop-blur">
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
        <Link href="/" className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-slate-900 text-white">
            <Sparkles size={18} />
          </div>

          <span className="text-lg font-bold tracking-tight text-slate-900">
            AFRICA-LMM
          </span>
        </Link>

        <nav className="flex items-center gap-1">
          {links.map((link) => {
            const Icon = link.icon;
            const active =
              link.href === "/"
                ? pathname === "/"
                : pathname.startsWith(link.href);

            return (
              <Link
                key={link.href}
                href={link.href}
                className={`flex items-center gap-2 rounded-lg px-3 py-2 text-sm transition ${
                  active
                    ? "bg-slate-900 text-white"
                    : "text-slate-600 hover:bg-slate-100 hover:text-slate-900"
                }`}
              >
                <Icon size={16} />
                <span className="hidden sm:inline">{link.label}</span>
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
