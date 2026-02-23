import { useLocation, useNavigate } from "react-router-dom";
import { ArrowLeft } from "lucide-react";
import OverallSimilarity from "../components/results/OverallSimilarity";
import BodyGroupChart from "../components/results/BodyGroupChart";
import VideoComparison from "../components/results/VideoComparison";
import CoachingFeedback from "../components/results/CoachingFeedback";
import type { AnalysisResponse } from "../types";

interface LocationState {
  result: AnalysisResponse;
  userVideoUrl: string;
}

export default function ResultPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const state = location.state as LocationState | null;

  if (!state?.result) {
    return (
      <div className="max-w-xl mx-auto px-6 py-20 text-center">
        <p className="text-surface-400 mb-4">No analysis results found.</p>
        <button
          onClick={() => navigate("/")}
          className="px-5 py-2.5 rounded-xl bg-court-600 text-white font-medium hover:bg-court-500 transition-colors"
        >
          Upload a Video
        </button>
      </div>
    );
  }

  const { result, userVideoUrl } = state;

  return (
    <div className="max-w-4xl mx-auto px-6 py-10">
      {/* Back button */}
      <button
        onClick={() => navigate("/")}
        className="flex items-center gap-2 text-surface-400 hover:text-white transition-colors mb-8 group"
      >
        <ArrowLeft className="w-4 h-4 group-hover:-translate-x-1 transition-transform" />
        <span className="text-sm">Analyze another swing</span>
      </button>

      <div className="space-y-8">
        <OverallSimilarity
          similarities={result.similarities}
          mostSimilarPlayer={result.most_similar_player}
        />

        <BodyGroupChart similarities={result.similarities} />

        <VideoComparison
          userVideoUrl={userVideoUrl}
          referenceVideoUrl={result.reference_video_url}
          mostSimilarPlayer={result.most_similar_player}
        />

        <CoachingFeedback coaching={result.coaching} />
      </div>
    </div>
  );
}
