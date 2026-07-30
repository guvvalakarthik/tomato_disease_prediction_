export default function SiteFooter() {
  return (
    <footer className="border-t border-slate-200 bg-white">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 py-8 flex flex-col sm:flex-row items-center justify-between gap-3 text-sm text-slate-500">
        <p>TomatoGuard ? uncertainty-aware tomato leaf screening</p>
        <div className="flex gap-4">
          <a className="hover:text-emerald-700" href="/model-card.html">Model card</a>
          <a className="hover:text-emerald-700" href="/field-evaluation.html">Field evaluation</a>
        </div>
      </div>
    </footer>
  );
}
