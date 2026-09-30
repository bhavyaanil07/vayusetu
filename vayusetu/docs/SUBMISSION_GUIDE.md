# Submission guide: step by step

The hackathon asks for five things. Here is how to produce each one from this kit.

| Requirement | What to use | Time |
|---|---|---|
| 1. GitHub repository link (public) | this `vayusetu` folder | 10 min |
| 2. Working prototype link | Streamlit Community Cloud (free) | 10 min |
| 3. Demo video link (public) | `VayuSetu_demo.mp4`, uploaded to YouTube or Google Drive | 10 min |
| 4. Presentation as PDF | the VayuSetu slide deck, exported to PDF | 5 min |
| 5. Brief description | `docs/brief_description.md` | 1 min |

---

## 1. Put the code on GitHub (public)

**Option A: terminal (recommended; works on your Linux laptop)**

1. Unzip `vayusetu.zip` and open a terminal inside the `vayusetu` folder.
2. On github.com, click **+ → New repository**. Name it `vayusetu`, choose **Public**, and do **not** add a README.
   Click **Create repository**.
3. Run these commands, replacing `YOUR-USERNAME`:

   ```bash
   git init
   git add .
   git commit -m "VayuSetu: federated clean-air intelligence prototype"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/vayusetu.git
   git push -u origin main
   ```

4. When asked for a password, paste a **Personal Access Token**, not your GitHub password. Create one at
   github.com → Settings → Developer settings → Personal access tokens → *Tokens (classic)* → Generate, and tick the
   `repo` scope. Alternatively, if you have GitHub CLI: `gh auth login`, then
   `gh repo create vayusetu --public --source=. --push`.

**Option B: browser upload (no git needed)**

1. Create the empty public repository as in step A2.
2. On the repository page, click **uploading an existing file**. Drag in **everything inside** the `vayusetu`
   folder (about 90 files).
3. Make sure the hidden `.streamlit` folder goes too. On Linux, press **Ctrl + H** in the file manager to show it.
4. Click **Commit changes**.

✅ Your repository link: `https://github.com/YOUR-USERNAME/vayusetu`

## 2. Get the working prototype link (Streamlit Community Cloud, free)

1. Go to <https://share.streamlit.io> and **sign in with GitHub**.
2. Click **Create app → Deploy a public app from GitHub**.
3. Fill in the form:
   - Repository: `YOUR-USERNAME/vayusetu`
   - Branch: `main`
   - Main file path: `app.py`
   - App URL: choose something like `vayusetu`
4. Open **Advanced settings** and set **Python version: 3.12**. Secrets are optional; see
   `.streamlit/secrets.toml.example`.
5. Click **Deploy**. The first build takes about 3–5 minutes.
6. When the app opens, click through every page once. Then switch the sidebar to **Live** to check that live
   Copernicus, weather and NASA fire data load. If a live service is down, the app falls back to replay by itself.

✅ Your prototype link: `https://vayusetu.streamlit.app` (or whatever URL you chose)

> Free Streamlit apps go to sleep after a few days without visitors. Open your link the day before judging,
> and click "Yes, get this app back up" if you see that message.

## 3. Upload the demo video (public link)

**YouTube**

1. Go to studio.youtube.com and click **Create → Upload videos**.
2. Choose `VayuSetu_demo.mp4` and set the title to *VayuSetu – Federated clean-air intelligence (demo)*.
3. Set visibility to **Unlisted** or **Public**.

**Google Drive**

1. Upload the file.
2. Right-click it → **Share** → General access: **Anyone with the link** (Viewer) → **Copy link**.

The provided video (about 4 minutes) is narrated, captioned and uses replay mode. For extra impact you can record
your own voice-over with `docs/demo_script.md`, ideally on the deployed app in Live mode.

## 4. Export the presentation as PDF

1. Open the VayuSetu pitch deck (16 slides, with speaker notes).
2. Fill in the placeholders: `[Team name]` on the first and last slides, and `[other team members]` plus the
   three links (prototype, video, GitHub) on the last slide. Click any text to edit it.
3. Export or download the deck as **PDF**.
4. Upload the PDF.

## 5. Brief description

Copy the short or full version from `docs/brief_description.md`.

---

## Final checklist

- [ ] GitHub repo is **public** and the README shows the screenshots
- [ ] README links updated: at the top of `README.md`, replace the two placeholder links with your prototype and video links
- [ ] Prototype link opens in a private/incognito window
- [ ] Video link plays in a private/incognito window
- [ ] Deck PDF uploaded
- [ ] Brief description pasted

## If something goes wrong

| Symptom | Fix |
|---|---|
| Streamlit build error mentioning Python | Set Python 3.12 in Advanced settings, then **Reboot app** |
| "ModuleNotFoundError" | Check that `requirements.txt` is in the repo root |
| Theme looks different | The `.streamlit/config.toml` file was not uploaded; add it |
| Live mode says "unavailable" | The external service is temporarily down; replay still works. Try again later |
| App slow on first load | Normal after sleeping: it loads models once, then it's fast |
