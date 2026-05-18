import { Trophy } from "lucide-react";
import type { PlayerSimilarity } from "../../types";

interface OverallSimilarityProps {
  similarities: PlayerSimilarity[];
  mostSimilarPlayer: string;
}

const PLAYER_COLORS: Record<string, { bg: string; border: string; text: string; bar: string }> = {
  Federer: { bg: "bg-federer/15", border: "border-federer/30", text: "text-federer", bar: "bg-federer" },
  Nadal: { bg: "bg-nadal/15", border: "border-nadal/30", text: "text-nadal", bar: "bg-nadal" },
  Djokovic: { bg: "bg-djokovic/15", border: "border-djokovic/30", text: "text-djokovic", bar: "bg-djokovic" },
  Murray: { bg: "bg-murray/15", border: "border-murray/30", text: "text-murray", bar: "bg-murray" },
};

export default function OverallSimilarity({
  similarities,
  mostSimilarPlayer,
}: OverallSimilarityProps) {
  const colors = PLAYER_COLORS[mostSimilarPlayer];
  const topMatch = similarities.find((s) => s.player === mostSimilarPlayer);

  return (
    <div className="space-y-6">
      {/* Hero card */}
      <div
        className={`rounded-2xl border p-6 text-center ${
          colors?.bg ?? "bg-surface-900"
        } ${colors?.border ?? "border-surface-700"}`}
      >
        <Trophy className={`w-8 h-8 mx-auto mb-2 ${colors?.text ?? "text-ball-400"}`} />
        <p className="text-surface-400 text-sm mb-1">Your swing is most similar to</p>
        <h2 className="font-display text-5xl tracking-wide text-white">
          {mostSimilarPlayer}
        </h2>
        <p className={`text-lg font-semibold mt-1 ${colors?.text ?? "text-court-400"}`}>
          {topMatch?.overall_similarity.toFixed(1)}% match
        </p>
      </div>

      {/* All players bar chart */}
      <div className="space-y-3">
        {similarities.map((s) => {
          const c = PLAYER_COLORS[s.player];
          return (
            <div key={s.player} className="flex items-center gap-4">
              <span className="w-20 text-sm text-surface-300 text-right font-medium">
                {s.player}
              </span>
              <div className="flex-1 h-8 bg-surface-800 rounded-lg overflow-hidden relative">
                <div
                  className={`h-full rounded-lg transition-all duration-1000 ease-out ${c?.bar ?? "bg-court-500"}`}
                  style={{ width: `${s.overall_similarity}%` }}
                />
                <span className="absolute right-3 top-1/2 -translate-y-1/2 text-xs font-semibold text-white/80">
                  {s.overall_similarity.toFixed(1)}%
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
