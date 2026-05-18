import { Play } from "lucide-react";

interface VideoComparisonProps {
  userVideoUrl: string | null;
  referenceVideoUrl: string | null;
  mostSimilarPlayer: string;
}

export default function VideoComparison({
  userVideoUrl,
  referenceVideoUrl,
  mostSimilarPlayer,
}: VideoComparisonProps) {
  return (
    <div className="rounded-2xl bg-surface-900 border border-surface-800 p-6">
      <h3 className="font-display text-2xl tracking-wide text-white mb-2">
        Side by Side
      </h3>
      <p className="text-surface-500 text-sm mb-6">
        Your swing compared to {mostSimilarPlayer}
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* User video */}
        <div>
          <p className="text-sm font-medium text-surface-400 mb-2">Your Swing</p>
          {userVideoUrl ? (
            <video
              src={userVideoUrl}
              className="w-full rounded-xl bg-black aspect-video object-contain"
              controls
              loop
              muted
            />
          ) : (
            <div className="w-full rounded-xl bg-surface-800 aspect-video flex items-center justify-center">
              <Play className="w-8 h-8 text-surface-600" />
            </div>
          )}
        </div>

        {/* Reference video */}
        <div>
          <p className="text-sm font-medium text-surface-400 mb-2">
            {mostSimilarPlayer}
          </p>
          {referenceVideoUrl ? (
            <video
              src={referenceVideoUrl}
              className="w-full rounded-xl bg-black aspect-video object-contain"
              controls
              loop
              muted
            />
          ) : (
            <div className="w-full rounded-xl bg-surface-800 aspect-video flex items-center justify-center">
              <p className="text-surface-500 text-sm">Reference video not available</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
