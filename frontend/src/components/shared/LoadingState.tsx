import { useState, useEffect } from "react";
import { Loader2, Check } from "lucide-react";

interface Step {
  label: string;
  duration: number;
}

const STEPS: Step[] = [
  { label: "Uploading video", duration: 2000 },
  { label: "Detecting body pose", duration: 8000 },
  { label: "Extracting biomechanics", duration: 3000 },
  { label: "Comparing to pro players", duration: 4000 },
  { label: "Generating feedback", duration: 2000 },
];

export default function LoadingState() {
  const [activeStep, setActiveStep] = useState(0);

  useEffect(() => {
    if (activeStep >= STEPS.length) return;

    const step = STEPS[activeStep];
    if (!step) return;

    const timer = setTimeout(() => {
      setActiveStep((prev) => Math.min(prev + 1, STEPS.length - 1));
    }, step.duration);

    return () => clearTimeout(timer);
  }, [activeStep]);

  return (
    <div className="flex flex-col items-center justify-center py-20 px-6">
      {/* Spinner */}
      <div className="relative mb-10">
        <div className="w-20 h-20 rounded-full border-2 border-surface-800" />
        <div className="absolute inset-0 w-20 h-20 rounded-full border-2 border-court-500 border-t-transparent animate-spin" />
      </div>

      {/* Steps */}
      <div className="w-full max-w-xs space-y-3">
        {STEPS.map((step, i) => {
          const isComplete = i < activeStep;
          const isActive = i === activeStep;

          return (
            <div
              key={step.label}
              className={`flex items-center gap-3 transition-all duration-500 ${
                isActive
                  ? "text-white"
                  : isComplete
                  ? "text-court-400"
                  : "text-surface-600"
              }`}
            >
              <div className="w-5 h-5 flex items-center justify-center flex-shrink-0">
                {isComplete ? (
                  <Check className="w-4 h-4" />
                ) : isActive ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <div className="w-1.5 h-1.5 rounded-full bg-current" />
                )}
              </div>
              <span className="text-sm">{step.label}</span>
            </div>
          );
        })}
      </div>

      <p className="text-surface-500 text-sm mt-10">
        This may take 1–3 minutes depending on video length.
      </p>
    </div>
  );
}
