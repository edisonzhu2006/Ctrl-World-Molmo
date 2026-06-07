(function () {
  const SESSION_KEY = "human_eval_session_id";
  const USER_KEY = "human_eval_user_id";

  const urlParams = new URLSearchParams(window.location.search);
  if (urlParams.get("user_id")) {
    localStorage.setItem(USER_KEY, urlParams.get("user_id"));
  }

  let sessionId = localStorage.getItem(SESSION_KEY);
  let userId = localStorage.getItem(USER_KEY);
  let currentItem = null;
  let itemStartedAt = null;
  let submitting = false;

  const appEl = document.getElementById("app");
  const doneEl = document.getElementById("done-screen");
  const progressText = document.getElementById("progress-text");
  const metaSample = document.getElementById("meta-sample");
  const metaPair = document.getElementById("meta-pair");
  const contextSection = document.getElementById("context-section");
  const contextContainer = document.getElementById("context-container");
  const gtVideo = document.getElementById("gt-video");
  const videoA = document.getElementById("video-a");
  const videoB = document.getElementById("video-b");
  const statusMsg = document.getElementById("status-msg");

  const allVideos = () => [gtVideo, videoA, videoB].filter(Boolean);

  function setStatus(msg, isError) {
    statusMsg.textContent = msg || "";
    statusMsg.classList.toggle("error", Boolean(isError));
  }

  function updateProgress(p) {
    if (!p) return;
    progressText.textContent = `${p.annotated} / ${p.total} annotated`;
  }

  function readOptionalInt(id) {
    const el = document.getElementById(id);
    if (!el || el.value === "") return null;
    const v = parseInt(el.value, 10);
    return Number.isFinite(v) ? v : null;
  }

  function clearOptionalFields() {
    ["visual-realism", "action-consistency", "temporal-coherence", "comments"].forEach((id) => {
      const el = document.getElementById(id);
      if (el) el.value = "";
    });
    document.querySelectorAll(".choice").forEach((b) => b.classList.remove("selected"));
  }

  function setupContext(item) {
    contextContainer.innerHTML = "";
    if (!item.context_url) {
      contextSection.classList.add("hidden");
      return;
    }
    contextSection.classList.remove("hidden");
    if (item.context_is_video) {
      const v = document.createElement("video");
      v.src = item.context_url;
      v.className = "vid";
      v.controls = true;
      v.playsInline = true;
      v.preload = "metadata";
      contextContainer.appendChild(v);
    } else {
      const img = document.createElement("img");
      img.src = item.context_url;
      img.className = "context-img";
      img.alt = "Context frame";
      contextContainer.appendChild(img);
    }
  }

  function loadItem(item) {
    currentItem = item;
    itemStartedAt = Date.now();
    clearOptionalFields();
    setStatus("");

    const parts = [];
    if (item.category) parts.push(item.category);
    if (item.sample_id) parts.push(item.sample_id);
    if (item.view_id) parts.push(item.view_id);
    metaSample.textContent = parts.join(" · ") || "";
    metaPair.textContent = `Pair ${item.pair_id}`;

    if (item.missing_files && item.missing_files.length) {
      setStatus(`Warning: missing files: ${item.missing_files.join(", ")}`, true);
    }

    setupContext(item);
    gtVideo.src = item.gt_url || "";
    videoA.src = item.video_a_url || "";
    videoB.src = item.video_b_url || "";

    allVideos().forEach((v) => {
      v.pause();
      v.currentTime = 0;
    });

    appEl.classList.remove("hidden");
    doneEl.classList.add("hidden");
  }

  async function fetchNext() {
    const params = new URLSearchParams();
    if (sessionId) params.set("session_id", sessionId);
    if (userId) params.set("user_id", userId);
    const res = await fetch(`/api/next?${params.toString()}`);
    if (!res.ok) throw new Error(`Failed to load next item (${res.status})`);
    return res.json();
  }

  async function loadNextItem() {
    setStatus("Loading…");
    const data = await fetchNext();
    sessionId = data.session_id;
    localStorage.setItem(SESSION_KEY, sessionId);
    if (data.user_id) {
      userId = data.user_id;
      localStorage.setItem(USER_KEY, userId);
    }
    updateProgress(data.progress);

    if (data.done) {
      appEl.classList.add("hidden");
      doneEl.classList.remove("hidden");
      setStatus("");
      return;
    }
    loadItem(data.item);
  }

  async function submitPreference(pref) {
    if (!currentItem || submitting) return;
    submitting = true;
    setStatus("Saving…");

    const spent = itemStartedAt ? (Date.now() - itemStartedAt) / 1000 : null;
    const body = {
      session_id: sessionId,
      user_id: userId,
      pair_id: currentItem.pair_id,
      preference: pref,
      time_spent_seconds: spent,
      visual_realism: readOptionalInt("visual-realism"),
      action_consistency: readOptionalInt("action-consistency"),
      temporal_coherence: readOptionalInt("temporal-coherence"),
      comments: document.getElementById("comments")?.value || null,
      browser_metadata: {
        userAgent: navigator.userAgent,
        language: navigator.language,
        viewport: `${window.innerWidth}x${window.innerHeight}`,
      },
    };

    try {
      const res = await fetch("/api/submit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Submit failed (${res.status})`);
      }
      const data = await res.json();
      updateProgress(data.progress);
      await loadNextItem();
    } catch (e) {
      setStatus(String(e.message || e), true);
    } finally {
      submitting = false;
    }
  }

  function syncPlay() {
    const videos = allVideos();
    const t = videos[0]?.currentTime || 0;
    videos.forEach((v) => {
      if (Math.abs(v.currentTime - t) > 0.05) v.currentTime = t;
    });
    videos.forEach((v) => v.play().catch(() => {}));
  }

  function syncPause() {
    allVideos().forEach((v) => v.pause());
  }

  function syncReplay() {
    allVideos().forEach((v) => {
      v.pause();
      v.currentTime = 0;
    });
    syncPlay();
  }

  function togglePlayPause() {
    const videos = allVideos();
    const anyPlaying = videos.some((v) => !v.paused);
    if (anyPlaying) syncPause();
    else syncPlay();
  }

  // Master sync: when GT seeks, mirror to A/B
  gtVideo.addEventListener("seeked", () => {
    const t = gtVideo.currentTime;
    [videoA, videoB].forEach((v) => {
      if (Math.abs(v.currentTime - t) > 0.05) v.currentTime = t;
    });
  });

  document.getElementById("btn-play-all")?.addEventListener("click", syncPlay);
  document.getElementById("btn-pause-all")?.addEventListener("click", syncPause);
  document.getElementById("btn-replay-all")?.addEventListener("click", syncReplay);

  document.querySelectorAll(".choice").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".choice").forEach((b) => b.classList.remove("selected"));
      btn.classList.add("selected");
      submitPreference(btn.dataset.pref);
    });
  });

  document.addEventListener("keydown", (e) => {
    if (e.target && ["INPUT", "TEXTAREA"].includes(e.target.tagName)) return;
    if (e.code === "Space") {
      e.preventDefault();
      togglePlayPause();
      return;
    }
    const map = { "1": "A", "2": "B", "3": "tie", "4": "invalid" };
    if (map[e.key]) {
      e.preventDefault();
      submitPreference(map[e.key]);
    }
  });

  loadNextItem().catch((e) => setStatus(String(e.message || e), true));
})();
