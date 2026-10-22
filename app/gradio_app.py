"""Веб-интерфейс прототипа (Gradio).

Запуск на компьютере:  python app/gradio_app.py --weights <папка с весами> [--examples <папка с примерами>] [--share]
В Colab интерфейс запускает блокнот 09.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from brainmri.serving import (to_gray_array, region_from_cam, traffic_light, render_overlay, to_rgb_uint8,
                              format_report, RU_NAMES, DISCLAIMER)

PRIVACY = ("**Данные.** При локальном запуске снимки не покидают компьютер. При запуске с публичной ссылкой (например, из Colab) "
           "данные проходят через сервис Gradio: используйте только обезличенные демонстрационные снимки. "
           "Приложение не ведёт журнал снимков; Gradio временно кэширует загруженные файлы во временной папке.")


def analyze_image(predictor, image):
    """Без зависимости от gradio (проверяется отдельно). Возвращает (снимок после обработки, карта внимания, словарь классов, текст)."""
    if image is None:
        return None, None, {}, "Загрузите снимок МРТ головного мозга (jpg или png)."
    try:
        gray = to_gray_array(image)
        r = predictor.predict(gray)
        light = traffic_light(r["conf"], r["dist"], predictor.t_conf, predictor.t_dist)
        show_region = r["classes"][r["pred"]] != "notumor"
        mask, _, frac = region_from_cam(r["cam"]) if show_region else (None, None, None)
        label = {RU_NAMES.get(c, c): float(p) for c, p in zip(r["classes"], r["probs"])}
        text = format_report(r["classes"], r["probs"], r["pred"], r["conf"], r["dist"], light, frac)
        return to_rgb_uint8(r["img"]), render_overlay(r["img"], r["cam"], mask), label, text
    except Exception as e:                      # некорректный файл, нестандартный формат и т.п.
        return None, None, {}, f"Не удалось обработать файл: {e}"


def _upload_only_image(gr, label):
    """Только загрузка файла: без веб-камеры и буфера обмена (фото экрана даёт нетипичное качество и лишние риски приватности)."""
    try:
        return gr.Image(type="numpy", sources=["upload"], label=label)        # Gradio 4 и новее
    except TypeError:
        return gr.Image(type="numpy", source="upload", label=label)           # Gradio 3


def build_demo(predictor, examples=None):
    import gradio as gr
    with gr.Blocks(title="МРТ головного мозга: прототип поддержки врача") as demo:
        gr.Markdown("# Прототип: поддержка анализа МРТ головного мозга\n" + DISCLAIMER)
        with gr.Row():
            with gr.Column():
                inp = _upload_only_image(gr, "Снимок МРТ (jpg или png)")
                btn = gr.Button("Проанализировать", variant="primary")
                if examples:
                    gr.Examples(examples=examples, inputs=inp, label="Примеры (выберите и нажмите «Проанализировать»)")
            with gr.Column():
                out_label = gr.Label(num_top_classes=4, label="Оценка модели по классам")
                out_text = gr.Markdown()
        with gr.Row():
            out_proc = gr.Image(label="Снимок после стандартизации (как его видит модель)")
            out_cam = gr.Image(label="Карта внимания и область интереса")
        gr.Markdown(PRIVACY)
        btn.click(lambda im: analyze_image(predictor, im), inputs=inp, outputs=[out_proc, out_cam, out_label, out_text])
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="папка с config.json и model_state.pt")
    ap.add_argument("--examples", default=None, help="папка с примерами снимков")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="публичная ссылка (только для обезличенных данных)")
    a = ap.parse_args()
    from brainmri.infer import Predictor
    predictor = Predictor(a.weights)
    ex = sorted(str(p) for p in Path(a.examples).glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}) if a.examples else None
    build_demo(predictor, ex).launch(server_name="127.0.0.1", server_port=a.port, share=a.share, show_api=False)


if __name__ == "__main__":
    main()
