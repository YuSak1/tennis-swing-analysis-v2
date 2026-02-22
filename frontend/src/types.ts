export interface PlayerSimilarity {
  player: string;
  overall_similarity: number;
  body_groups: Record<string, number>;
}

export interface CoachingTip {
  type: "strength" | "improvement";
  body_part: string;
  message: string;
}

export interface SwingPhases {
  preparation: [number, number];
  forward_swing: [number, number];
  contact: number;
  follow_through: [number, number];
  recovery: [number, number];
}

export interface AnalysisResponse {
  most_similar_player: string;
  similarities: PlayerSimilarity[];
  coaching: CoachingTip[];
  phases: SwingPhases;
  landmarks: (Record<string, number>[] | null)[] | null;
  reference_video_url: string | null;
}

export type Hand = "right" | "left";
