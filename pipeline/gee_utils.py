"""Earth Engine helpers shared by 00a, 02a and 04. Work the same locally and in Google Colab.

Settings come from environment variables, so a Colab cell can set them once:
  GEE_PROJECT        Cloud project registered for Earth Engine
  AMA_AOI_ASSET      AMA polygon asset (empty -> the script's placeholder bounding box)
  GEE_DRIVE_FOLDER   Drive folder for the exports of the script being run
  GEE_WAIT=1         block until the export tasks finish (so the next cell can read the files)
"""
import os
import time

AUTH_HELP = ("Earth Engine is not authenticated or the project is wrong.\n"
             "Colab: run once in a notebook cell\n"
             "    import ee; ee.Authenticate(); ee.Initialize(project='<your-project>')\n"
             "and start the GEE scripts with %run so they share that session.\n"
             "Local: run `earthengine authenticate` once.")


def init_ee(project=None):
    import ee
    project = project or os.environ.get("GEE_PROJECT") or None
    try:
        ee.Initialize(project=project)
    except Exception as exc:                      # the message from ee alone rarely says what to do
        raise RuntimeError(AUTH_HELP) from exc
    return ee


def aoi_geometry(ee, default_bbox):
    asset = os.environ.get("AMA_AOI_ASSET", "").strip()
    return ee.FeatureCollection(asset).geometry() if asset else ee.Geometry.Rectangle(default_bbox)


def wait_requested():
    return os.environ.get("GEE_WAIT", "0") == "1"


def wait_tasks(tasks, poll_s=30, timeout_s=6 * 3600, log=print, sleep=time.sleep):
    """Poll export tasks until all finish. Raises if any fails or the timeout passes."""
    pending, failed, t0 = list(tasks), [], time.time()
    while pending:
        still = []
        for t in pending:
            st = t.status()
            state, name = st.get("state"), st.get("description", "?")
            if state == "COMPLETED":
                log(f"done: {name}")
            elif state in ("FAILED", "CANCELLED"):
                failed.append(name)
                log(f"{state}: {name} {st.get('error_message', '')}")
            else:
                still.append(t)
        pending = still
        if pending:
            if time.time() - t0 > timeout_s:
                raise TimeoutError(f"{len(pending)} export task(s) still running after {timeout_s / 3600:.1f} h")
            log(f"waiting for {len(pending)} task(s)...")
            sleep(poll_s)
    if failed:
        raise RuntimeError(f"export failed: {', '.join(failed)} (see https://code.earthengine.google.com/tasks)")
