export const DISEASES = {
  tomato_bacterial_spot: {
    name: "Bacterial Spot",
    severity: "high",
    emoji: "BS",
    description:
      "Often appears as small water-soaked spots that darken on leaves or fruit.",
    guidance:
      "Isolate affected material, keep foliage dry, and ask a local agricultural specialist to confirm the cause and region-appropriate management.",
  },
  tomato_early_blight: {
    name: "Early Blight",
    severity: "medium",
    emoji: "EB",
    description:
      "Often produces dark concentric-ring lesions on older leaves, sometimes with yellowing.",
    guidance:
      "Remove heavily affected debris, improve airflow, water at soil level, and obtain local expert guidance before applying a treatment.",
  },
  tomato_late_blight: {
    name: "Late Blight",
    severity: "high",
    emoji: "LB",
    description:
      "Can produce rapidly spreading gray-green or brown lesions during cool, wet conditions.",
    guidance:
      "Separate suspicious plants promptly and seek urgent confirmation from a local extension or plant-health professional.",
  },
  tomato_leaf_mold: {
    name: "Leaf Mold",
    severity: "medium",
    emoji: "LM",
    description:
      "May show pale upper-leaf patches with olive-colored growth underneath, especially in humid spaces.",
    guidance:
      "Reduce humidity and improve ventilation while arranging a qualified assessment.",
  },
  tomato_septoria_leaf_spot: {
    name: "Septoria Leaf Spot",
    severity: "medium",
    emoji: "SL",
    description:
      "Often creates many small circular spots with darker borders, beginning on lower leaves.",
    guidance:
      "Remove affected fallen material, avoid wetting foliage, and confirm management with a local specialist.",
  },
  tomato_spider_mites: {
    name: "Spider Mites",
    severity: "medium",
    emoji: "SM",
    description:
      "Feeding damage can cause fine stippling, yellowing, and webbing on leaf undersides.",
    guidance:
      "Inspect leaf undersides and obtain local integrated-pest-management advice before choosing a control.",
  },
  tomato_target_spot: {
    name: "Target Spot",
    severity: "medium",
    emoji: "TS",
    description:
      "May cause brown lesions with ring-like patterns on leaves, stems, or fruit.",
    guidance:
      "Improve airflow, reduce prolonged leaf wetness, and have the symptoms confirmed locally.",
  },
  tomato_yellow_leaf_curl_virus: {
    name: "Yellow Leaf Curl Virus",
    severity: "high",
    emoji: "YC",
    description:
      "Typical symptoms can include upward curling, yellowing, and stunted growth.",
    guidance:
      "Separate suspicious plants and consult a local expert about confirmation and vector management.",
  },
  tomato_mosaic_virus: {
    name: "Mosaic Virus",
    severity: "high",
    emoji: "MV",
    description:
      "Can cause mottled light and dark green patterns, distortion, and reduced vigor.",
    guidance:
      "Avoid handling other plants after contact, clean tools, and request expert confirmation.",
  },
  tomato_healthy: {
    name: "Healthy",
    severity: "none",
    emoji: "OK",
    description:
      "The model did not identify one of its nine supported disease or pest classes.",
    guidance:
      "Continue routine monitoring. A healthy prediction does not rule out unsupported conditions or early-stage symptoms.",
  },
};

export const SEVERITY_STYLES = {
  high: "bg-red-100 text-red-700 border-red-200",
  medium: "bg-amber-100 text-amber-700 border-amber-200",
  none: "bg-emerald-100 text-emerald-700 border-emerald-200",
};
