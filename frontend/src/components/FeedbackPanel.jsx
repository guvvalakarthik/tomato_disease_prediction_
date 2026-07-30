import { useState } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function FeedbackPanel() {
  const [submitted, setSubmitted] = useState(false);
  const [error, setError] = useState("");

  async function submit(event) {
    event.preventDefault();
    setError("");
    const form = new FormData(event.currentTarget);
    const payload = {
      consent: form.get("consent") === "on",
      participant_role: form.get("participant_role"),
      task_completed: form.get("task_completed") === "on",
      interpretation_without_help: form.get("interpretation_without_help") === "on",
      uncertainty_understood: form.get("uncertainty_understood") === "on",
      expert_confirmation_intended: form.get("expert_confirmation_intended") === "on",
      usefulness: Number(form.get("usefulness")),
      clarity: Number(form.get("clarity")),
      issue_tags: form.getAll("issue_tags"),
    };
    try {
      const response = await fetch(`${API_URL}/v1/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!response.ok) throw new Error("Feedback could not be saved.");
      setSubmitted(true);
    } catch (requestError) {
      setError(requestError.message);
    }
  }

  if (submitted) {
    return <p className="mt-5 text-sm text-emerald-700">Thank you. Your anonymous study response was recorded.</p>;
  }

  return (
    <details className="mt-5 border-t border-slate-200 pt-4">
      <summary className="cursor-pointer text-sm font-semibold text-slate-700">
        Optional anonymous usability study
      </summary>
      <form className="mt-4 space-y-3 text-sm text-slate-700" onSubmit={submit}>
        <p>No image, filename, IP address, contact information, or free text is collected.</p>
        <label className="block">
          Your role
          <select name="participant_role" required className="ml-2 border rounded p-1">
            <option value="farmer">Farmer</option>
            <option value="agriculture_student">Agriculture student</option>
            <option value="domain_reviewer">Plant-health reviewer</option>
            <option value="other">Other</option>
          </select>
        </label>
        {[
          ["task_completed", "I completed upload-to-interpretation."],
          ["interpretation_without_help", "I interpreted the result without help."],
          ["uncertainty_understood", "I understand what an uncertain result means."],
          ["expert_confirmation_intended", "I would seek expert confirmation before treatment."],
        ].map(([name, label]) => (
          <label key={name} className="block"><input type="checkbox" name={name} className="mr-2" />{label}</label>
        ))}
        <label className="block">Usefulness (1?5) <input name="usefulness" type="number" min="1" max="5" required className="ml-2 w-16 border rounded p-1" /></label>
        <label className="block">Clarity (1?5) <input name="clarity" type="number" min="1" max="5" required className="ml-2 w-16 border rounded p-1" /></label>
        <fieldset>
          <legend>Issues encountered (optional)</legend>
          {[
            ["upload_difficult", "Upload was difficult"],
            ["result_unclear", "Result was unclear"],
            ["confidence_misleading", "Confidence was misleading"],
            ["uncertainty_unclear", "Uncertainty was unclear"],
            ["attention_map_unclear", "Attention map was unclear"],
            ["accessibility_problem", "Accessibility problem"],
          ].map(([value, label]) => (
            <label key={value} className="block"><input type="checkbox" name="issue_tags" value={value} className="mr-2" />{label}</label>
          ))}
        </fieldset>
        <label className="block font-medium"><input required type="checkbox" name="consent" className="mr-2" />I consent to these anonymous categorical responses being used for the TomatoGuard usability study.</label>
        <button type="submit" className="rounded-lg bg-emerald-700 px-4 py-2 text-white">Submit anonymous feedback</button>
        {error && <p role="alert" className="text-red-700">{error}</p>}
      </form>
    </details>
  );
}
