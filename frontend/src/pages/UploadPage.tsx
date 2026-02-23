import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Info } from "lucide-react";
import VideoUploader from "../components/upload/VideoUploader";
import AnalysisOptions from "../components/upload/AnalysisOptions";
import LoadingState from "../components/shared/LoadingState";
import ErrorMessage from "../components/shared/ErrorMessage";
import { analyzeSwing } from "../api/client";
import type { Hand } from "../types";

export default function UploadPage() {
  const navigate = useNavigate();
  const [file, setFile] = useState<File | null>(null);
  const [hand, setHand] = useState<Hand>("right");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async () => {
    if (!file) return;

    setIsLoading(true);
    setError(null);

    try {
      const result = await analyzeSwing(file, hand);
      const videoUrl = URL.createObjectURL(file);
      navigate("/results", { state: { result, userVideoUrl: videoUrl } });
    } catch (err: unknown) {
      let message = "Something went wrong. Please try again.";
      if (err instanceof Error) {
        message = err.message;
      }
      if (
        typeof err === "object" &&
        err !== null &&
        "response" in err
      ) {
        const axiosErr = err as { response?: { data?: { detail?: string } } };
        message = axiosErr.response?.data?.detail ?? message;
      }
      setError(message);
    } finally {
      setIsLoading(false);
    }
  };

  if (isLoading) {
    return <LoadingState />;
  }

  return (
    <div className="max-w-xl mx-auto px-6 py-12">
      {/* Hero */}
      <div className="text-center mb-10">
        <h2 className="font-display text-5xl md:text-6xl tracking-wide text-white mb-3">
          Analyze Your Forehand
        </h2>
        <p className="text-surface-400 text-lg max-w-md mx-auto">
          Upload a video of your swing and see how it compares to Federer,
          Nadal, Djokovic, and Murray.
        </p>
      </div>

      {/* Upload + Options */}
      <div className="space-y-6">
        <VideoUploader file={file} onFileChange={setFile} />

        {error && (
          <ErrorMessage message={error} onRetry={() => setError(null)} />
        )}

        <AnalysisOptions
          hand={hand}
          onHandChange={setHand}
          onSubmit={handleSubmit}
          isDisabled={!file}
          isLoading={isLoading}
        />

        {/* Requirements */}
        <div className="rounded-xl bg-surface-900/50 border border-surface-800 p-5">
          <div className="flex items-start gap-3">
            <Info className="w-4 h-4 text-surface-500 mt-0.5 flex-shrink-0" />
            <div className="text-sm text-surface-500 space-y-1">
              <p className="text-surface-400 font-medium">Video requirements</p>
              <p>Only forehand shots — no backhand, serve, or other shots.</p>
              <p>Court-level angle, filmed from behind the player.</p>
              <p>Full body visible. At least 480p quality.</p>
              <p>Only the first 10 seconds will be analyzed.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
