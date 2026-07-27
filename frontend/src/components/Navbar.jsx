import { Leaf } from "lucide-react";

export default function Navbar() {
  return (
    <header className="fixed top-0 inset-x-0 z-50 bg-white/70 backdrop-blur-md border-b border-emerald-100">
      <nav className="max-w-6xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between">
        <a href="#" className="flex items-center gap-2 font-bold text-lg text-emerald-900">
          <span className="grid place-items-center w-9 h-9 rounded-xl bg-emerald-600 text-white">
            <Leaf size={18} />
          </span>
          TomatoGuard
        </a>
        <div className="flex items-center gap-6 text-sm font-medium text-emerald-800">
          <a href="#detect" className="hover:text-emerald-600 transition-colors">Detect</a>
          <a href="#diseases" className="hidden sm:block hover:text-emerald-600 transition-colors">Disease Library</a>
          <a
            href="#detect"
            className="px-4 py-2 rounded-full bg-emerald-600 text-white hover:bg-emerald-700 transition-colors"
          >
            Try Now
          </a>
        </div>
      </nav>
    </header>
  );
}
