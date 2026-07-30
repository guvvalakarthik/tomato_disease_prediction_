# TomatoGuard web client

Vite, React 19, and Tailwind CSS client for the versioned TomatoGuard API.

```bash
npm ci
npm run dev
npm test
npm run build
```

Set `VITE_API_URL` at build time; local development defaults to
`http://127.0.0.1:8000`.

The client:

- accepts JPEG/PNG files up to 4 MB (below Vercel's 4.5 MB body limit);
- displays `uncertain` as a refusal, never as a diagnosis;
- shows top-three calibrated probabilities and model version;
- displays a class-activation map when the release supports it;
- uses cautious, non-prescriptive guidance and a visible disclaimer;
- explains free-tier cold starts without exposing server details.

The production container uses an unprivileged Nginx process on port 8080. For a
static deployment, build `dist/` and publish it with the supplied Cloudflare
Pages configuration.
