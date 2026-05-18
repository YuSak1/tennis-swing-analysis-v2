import { Link } from "react-router-dom";
import { Activity } from "lucide-react";

export default function Header() {
  return (
    <header className="border-b border-surface-800">
      <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
        <Link to="/" className="flex items-center gap-3 group">
          <div className="w-9 h-9 rounded-lg bg-court-600 flex items-center justify-center group-hover:bg-court-500 transition-colors">
            <Activity className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="font-display text-2xl tracking-wide text-white leading-none">
              Swing Analysis
            </h1>
            <p className="text-[11px] text-surface-400 tracking-widest uppercase">
              Tennis Forehand
            </p>
          </div>
        </Link>

        <a
          href="https://github.com/YuSak1/tennis-swing-analysis-v2"
          target="_blank"
          rel="noopener noreferrer"
          className="text-surface-500 hover:text-surface-300 transition-colors text-sm"
        >
          GitHub
        </a>
      </div>
    </header>
  );
}
