import { Hand as HandIcon } from "lucide-react";
import type { Hand } from "../../types";

interface AnalysisOptionsProps {
  hand: Hand;
  onHandChange: (hand: Hand) => void;
  onSubmit: () => void;
  isDisabled: boolean;
  isLoading: boolean;
}

const HAND_OPTIONS: { value: Hand; label: string }[] = [
  { value: "right", label: "Right-handed" },
  { value: "left", label: "Left-handed" },
];

export default function AnalysisOptions({
  hand,
  onHandChange,
  onSubmit,
  isDisabled,
  isLoading,
}: AnalysisOptionsProps) {
  return (
    <div className="space-y-6">
      {/* Hand selection */}
      <div>
        <label className="block text-sm font-medium text-surface-300 mb-3">
          <HandIcon className="w-4 h-4 inline mr-2 -mt-0.5" />
          Dominant Hand
        </label>
        <div className="flex gap-3">
          {HAND_OPTIONS.map((opt) => (
            <button
              key={opt.value}
              onClick={() => onHandChange(opt.value)}
              className={`
                flex-1 py-2.5 px-4 rounded-xl text-sm font-medium
                transition-all duration-200 border
                ${
                  hand === opt.value
                    ? "bg-court-600 border-court-500 text-white"
                    : "bg-surface-900 border-surface-700 text-surface-300 hover:border-surface-500"
                }
              `}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      {/* Submit */}
      <button
        onClick={onSubmit}
        disabled={isDisabled || isLoading}
        className={`
          w-full py-3.5 rounded-xl font-semibold text-base
          transition-all duration-200
          ${
            isDisabled || isLoading
              ? "bg-surface-800 text-surface-500 cursor-not-allowed"
              : "bg-court-600 text-white hover:bg-court-500 active:scale-[0.98] shadow-lg shadow-court-600/20"
          }
        `}
      >
        {isLoading ? "Analyzing..." : "Analyze My Swing"}
      </button>
    </div>
  );
}
