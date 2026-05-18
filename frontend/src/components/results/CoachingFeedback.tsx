import { CheckCircle2, Lightbulb } from "lucide-react";
import type { CoachingTip } from "../../types";

interface CoachingFeedbackProps {
  coaching: CoachingTip[];
}

export default function CoachingFeedback({ coaching }: CoachingFeedbackProps) {
  if (coaching.length === 0) return null;

  const improvements = coaching.filter((t) => t.type === "improvement");
  const strengths = coaching.filter((t) => t.type === "strength");

  return (
    <div className="rounded-2xl bg-surface-900 border border-surface-800 p-6">
      <h3 className="font-display text-2xl tracking-wide text-white mb-6">
        Coaching Feedback
      </h3>

      {/* Improvements */}
      {improvements.length > 0 && (
        <div className="mb-6">
          <h4 className="text-sm font-semibold text-ball-400 uppercase tracking-wider mb-3">
            Areas to Improve
          </h4>
          <div className="space-y-3">
            {improvements.map((tip, i) => (
              <div
                key={i}
                className="flex gap-3 p-4 rounded-xl bg-ball-500/5 border border-ball-500/10"
              >
                <Lightbulb className="w-5 h-5 text-ball-400 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-sm font-medium text-ball-400 mb-0.5">
                    {tip.body_part}
                  </p>
                  <p className="text-sm text-surface-300 leading-relaxed">
                    {tip.message}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Strengths */}
      {strengths.length > 0 && (
        <div>
          <h4 className="text-sm font-semibold text-court-400 uppercase tracking-wider mb-3">
            Strengths
          </h4>
          <div className="space-y-3">
            {strengths.map((tip, i) => (
              <div
                key={i}
                className="flex gap-3 p-4 rounded-xl bg-court-500/5 border border-court-500/10"
              >
                <CheckCircle2 className="w-5 h-5 text-court-400 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-sm font-medium text-court-400 mb-0.5">
                    {tip.body_part}
                  </p>
                  <p className="text-sm text-surface-300 leading-relaxed">
                    {tip.message}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
