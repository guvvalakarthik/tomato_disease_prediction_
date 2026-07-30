import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import PredictionResult from "./PredictionResult.jsx";

const model = { version: "1.0.0", sha256: "abc" };

describe("PredictionResult", () => {
  it("does not turn an uncertain response into a diagnosis", () => {
    render(
      <PredictionResult
        result={{
          status: "uncertain",
          model,
          prediction: null,
          uncertainty_reason: "Below threshold",
          top_predictions: [
            {
              class_id: "tomato_early_blight",
              label: "Early Blight",
              probability: 0.31,
            },
          ],
          explanation: null,
          disclaimer: "Educational screening only.",
        }}
      />,
    );

    expect(screen.getByText("Unable to diagnose confidently")).toBeInTheDocument();
    expect(screen.getByText("Below threshold")).toBeInTheDocument();
    expect(screen.queryByText("General next step")).not.toBeInTheDocument();
  });

  it("shows calibrated output, safe guidance, and model identity", () => {
    render(
      <PredictionResult
        result={{
          status: "predicted",
          model,
          prediction: {
            class_id: "tomato_early_blight",
            label: "Early Blight",
            probability: 0.93,
          },
          uncertainty_reason: null,
          top_predictions: [
            {
              class_id: "tomato_early_blight",
              label: "Early Blight",
              probability: 0.93,
            },
          ],
          explanation: null,
          disclaimer: "Educational screening only.",
        }}
      />,
    );

    expect(screen.getAllByText("Early Blight").length).toBeGreaterThan(0);
    expect(screen.getAllByText("93.0%")).toHaveLength(2);
    expect(screen.getByText("General next step")).toBeInTheDocument();
    expect(screen.getByText("Model 1.0.0")).toBeInTheDocument();
  });
});
