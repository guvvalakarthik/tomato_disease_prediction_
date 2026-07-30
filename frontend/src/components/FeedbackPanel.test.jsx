import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import FeedbackPanel from "./FeedbackPanel.jsx";


describe("FeedbackPanel", () => {
  afterEach(() => vi.restoreAllMocks());

  it("submits only consented categorical feedback", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue({ ok: true });
    render(<FeedbackPanel />);
    fireEvent.click(screen.getByText("Optional anonymous usability study"));
    fireEvent.change(screen.getByLabelText(/Usefulness/), { target: { value: "4" } });
    fireEvent.change(screen.getByLabelText(/Clarity/), { target: { value: "5" } });
    fireEvent.click(screen.getByLabelText(/I completed upload-to-interpretation/));
    fireEvent.click(screen.getByLabelText(/I consent to these anonymous/));
    fireEvent.click(screen.getByRole("button", { name: "Submit anonymous feedback" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledOnce());
    const payload = JSON.parse(fetchMock.mock.calls[0][1].body);
    expect(payload).toMatchObject({
      consent: true,
      participant_role: "farmer",
      task_completed: true,
      usefulness: 4,
      clarity: 5,
    });
    expect(payload).not.toHaveProperty("comment");
    expect(await screen.findByText(/anonymous study response was recorded/)).toBeTruthy();
  });

  it("shows a bounded error when storage fails", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({ ok: false });
    render(<FeedbackPanel />);
    fireEvent.click(screen.getByText("Optional anonymous usability study"));
    fireEvent.change(screen.getByLabelText(/Usefulness/), { target: { value: "3" } });
    fireEvent.change(screen.getByLabelText(/Clarity/), { target: { value: "3" } });
    fireEvent.click(screen.getByLabelText(/I consent to these anonymous/));
    fireEvent.click(screen.getByRole("button", { name: "Submit anonymous feedback" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Feedback could not be saved.");
  });
});
