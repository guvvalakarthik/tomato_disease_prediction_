export const DISEASES = {
  Tomato_Bacterial_spot: {
    name: "Bacterial Spot",
    severity: "high",
    emoji: "🦠",
    description:
      "Caused by Xanthomonas bacteria. Small, water-soaked spots on leaves and fruit that turn dark brown with a greasy appearance.",
    treatment:
      "Remove infected plants, avoid overhead watering, and apply copper-based bactericides early. Rotate crops for 2–3 years.",
  },
  Tomato_Early_blight: {
    name: "Early Blight",
    severity: "medium",
    emoji: "🍂",
    description:
      "Fungal disease (Alternaria solani) producing dark concentric ring spots on older leaves, often with yellow halos.",
    treatment:
      "Prune lower leaves, mulch the soil, improve air circulation, and apply fungicides like chlorothalonil or mancozeb.",
  },
  Tomato_Late_blight: {
    name: "Late Blight",
    severity: "high",
    emoji: "🌧️",
    description:
      "Devastating disease caused by Phytophthora infestans. Large, greasy gray-green lesions that spread rapidly in cool wet weather.",
    treatment:
      "Destroy infected plants immediately. Apply preventive fungicides and choose resistant varieties. Avoid wet foliage overnight.",
  },
  Tomato_Leaf_Mold: {
    name: "Leaf Mold",
    severity: "medium",
    emoji: "🍄",
    description:
      "Caused by Passalora fulva, common in greenhouses. Pale yellow spots on upper leaf surfaces with olive-green mold beneath.",
    treatment:
      "Reduce humidity below 85%, increase ventilation, remove infected leaves, and use resistant cultivars.",
  },
  Tomato_Septoria_leaf_spot: {
    name: "Septoria Leaf Spot",
    severity: "medium",
    emoji: "🔵",
    description:
      "Fungal disease creating many small circular spots with dark borders and gray centers, starting on lower leaves.",
    treatment:
      "Remove infected foliage, mulch around plants, water at the base, and apply fungicide at first sign of spots.",
  },
  Tomato_Spider_mites: {
    name: "Spider Mites",
    severity: "medium",
    emoji: "🕷️",
    description:
      "Two-spotted spider mites feed on leaf undersides causing stippling, yellowing, and fine webbing in heavy infestations.",
    treatment:
      "Spray plants with water, apply insecticidal soap or neem oil, and encourage predatory mites. Keep plants well-watered.",
  },
  Tomato_Target_Spot: {
    name: "Target Spot",
    severity: "medium",
    emoji: "🎯",
    description:
      "Caused by Corynespora cassiicola. Brown lesions with concentric rings resembling a target, affecting leaves, stems and fruit.",
    treatment:
      "Improve airflow, avoid leaf wetness, remove crop debris, and apply protective fungicides regularly.",
  },
  Tomato_Yellow_Leaf_Curl_Virus: {
    name: "Yellow Leaf Curl Virus",
    severity: "high",
    emoji: "🌀",
    description:
      "Whitefly-transmitted virus causing upward leaf curling, yellowing, stunted growth, and severe yield loss.",
    treatment:
      "No cure — remove infected plants. Control whiteflies with sticky traps and insecticides. Use resistant varieties.",
  },
  Tomato_mosaic_virus: {
    name: "Mosaic Virus",
    severity: "high",
    emoji: "🧩",
    description:
      "Highly contagious virus causing mottled light/dark green mosaic patterns, distorted leaves, and reduced fruit quality.",
    treatment:
      "No chemical cure. Remove and destroy infected plants, disinfect tools, wash hands, and plant resistant varieties.",
  },
  Tomato_healthy: {
    name: "Healthy",
    severity: "none",
    emoji: "✅",
    description:
      "No disease detected! Your tomato leaf shows healthy green tissue with no visible signs of infection or pest damage.",
    treatment:
      "Keep up the good work — water consistently, fertilize appropriately, and monitor regularly for early signs of disease.",
  },
};

export const SEVERITY_STYLES = {
  high: "bg-red-100 text-red-700 border-red-200",
  medium: "bg-amber-100 text-amber-700 border-amber-200",
  none: "bg-emerald-100 text-emerald-700 border-emerald-200",
};
