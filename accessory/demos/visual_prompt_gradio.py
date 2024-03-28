import json
import torch
import numpy as np
from typing import List, Optional
import argparse
import multiprocessing as mp
import gradio as gr
import math

import regex as re
from PIL import Image, ImageDraw, ImageFont
from segment_anything import sam_model_registry, SamPredictor

class Ready: pass
class ModelFailure: pass


def extract_masks_from_draw(img_input_draw):
    masks_obj = []
    # print(img_input_draw['layers'])
    for p in img_input_draw['layers']:
        img_array = np.array(p)

        rows, cols = np.where(np.sum(img_array, axis=-1) != 0)
        
        if len(rows) == 0 or len(cols) == 0:
            continue
        
        minx, maxx = np.min(cols), np.max(cols)
        miny, maxy = np.min(rows), np.max(rows)
        masks_obj.append([minx, miny, maxx, maxy])
    return masks_obj


def on_draw(evt: gr.SelectData, vtype, masks):
    # print(evt.__dict__)
    x = evt.index[0]
    y = evt.index[1]
    
    masks_obj = json.loads(masks)["mask"]
    if vtype == "Point":
        masks_obj.append((x, y))
    elif vtype == "Box":
        if len(masks_obj) == 0 or len(masks_obj[-1]) == 4:
            masks_obj.append((x, y))
        else:
            x0 = masks_obj[-1][0]
            y0 = masks_obj[-1][1]
            if x < x0:
                x0, x = x, x0
            if y < y0:
                y0, y = y, y0

            masks_obj[-1] = (x0, y0, x, y)
    else:
        # draw
        raise "on_draw shouldn't be called for Free-Form Draw model"


    return json.dumps({"mask": masks_obj})


def draw_masks(vtype, img, masks):
    if img==None:
        return None
    
    width, height = img.size
    min_dimension = min(width, height)
    if min_dimension <= 256:
        font_size = 8
        r = 4
        width = 3
    elif min_dimension <= 512:
        font_size = 16
        r = 8
        width = 4
    elif min_dimension <= 1024:
        font_size = 32
        r = 16
        width = 6
    elif min_dimension <= 1536:
        font_size = 60
        r = 30
        width = 7
    elif min_dimension <= 2048:
        font_size = 80
        r = 40
        width = 8
    else:
        font_size = 110
        r = 55
        width = 9

    font = ImageFont.truetype("times.ttf", font_size)
    mask_id = 1
    out_img = img.copy()
    draw = ImageDraw.Draw(out_img)
    for mask in masks:
        if vtype == "Point":
            x, y = mask
            draw.ellipse((x-r, y-r, x+r, y+r), fill='red', outline='red')
            draw.text((x-0.5*r, y-r), text=str(mask_id),fill="white", font=font)
        elif vtype == "Box":
            if len(mask) == 4:
                x0, y0, x, y = mask
                draw.rectangle((x0, y0, x, y), outline="red", width=width)
                _, _, w, h = draw.textbbox((0, 0), text=str(mask_id), font=font)
                # draw.text((x0+(x-x0-w)/2, y0+(y-y0-h)/2), text=str(mask_id), font=font, fill="white")
                bg_x0 = x0 + (x - x0 - w) / 2
                bg_y0 = y0 + (y - y0 - h) / 2
                bg_x = bg_x0 + w
                bg_y = bg_y0 + h + 1
                draw.rectangle([bg_x0, bg_y0, bg_x, bg_y], fill="black")
                draw.text((bg_x0, bg_y0), text=str(mask_id), font=font, fill="white")
            else:
                x,y = mask
                r_tmp = r//2
                draw.ellipse((x-r_tmp, y-r_tmp, x+r_tmp, y+r_tmp), fill='red', outline='red')
        mask_id += 1
    return out_img


def show_box(img: Image, box, color):
    img_box = img.copy()
    draw = ImageDraw.Draw(img_box)

    x0, y0 = box[0], box[1]
    x1, y1 = box[2], box[3]

    draw.rectangle((x0, y0, x1, y1), outline=color, width=3)

    return img_box 


def draw_star(draw, center, diameter, color):
    angles = [72 * i for i in range(5)]

    outer_points = [(center[0] + diameter * 0.5 * math.sin(math.radians(angle)),
                     center[1] - diameter * 0.5 * math.cos(math.radians(angle))) for angle in angles]

    inner_diameter = diameter * 0.382
    inner_angles = [72 * i + 36 for i in range(5)] 
    inner_points = [(center[0] + inner_diameter * 0.5 * math.sin(math.radians(angle)),
                     center[1] - inner_diameter * 0.5 * math.cos(math.radians(angle))) for angle in inner_angles]

    star_points = [None]*(len(outer_points)+len(inner_points))
    star_points[::2] = outer_points
    star_points[1::2] = inner_points

    draw.polygon(star_points, fill=color)

def show_mask(img: Image, mask, color):
        
    alpha_value = int(0.5 * 255)
    alpha_mask = (mask * alpha_value).byte()
    mask_img = Image.new("RGB", img.size, color)

    blended_img = Image.composite(mask_img, img, Image.fromarray(alpha_mask.cpu().numpy(), "L"))
    return blended_img

def show_point(img: Image, point, color, diameter=25):
    img_point = img.copy()
    draw = ImageDraw.Draw(img_point)

    draw_star(draw, point, diameter, (63,87,42))

    return img_point

def draw_mask_on_image(img: Image, vp, predictor):
    # Red (255,0,0), Orange (255,165,0), Yellow (255,255,0), Green (0,128,0), 
    # Cyan (0,255,255), Blue (0,0,255), Purple (128,0,128)
    # Magenta/Fuchsia (255,0,255), Lime (0,255,0), and Teal (0,128,128)
    colors_list = [(255,0,0), (255,165,0), (255,255,0), (0,128,0), (0,255,255), (0,0,255), (128,0,128), (255,0,255), (0,255,0), (0,128,128)]
    mask_colors = colors_list[:len(vp)]

    if len(vp[0]) == 2:
        label = "Point"
    elif len(vp[0]) == 4:
        label = "Box"

    if len(vp) > 0:
        img_mask = img.copy()
        img_array = np.array(img)
        predictor.set_image(img_array)
        if label == "Point":
            input_points = torch.tensor(vp, device=predictor.device)
            input_labels = torch.ones(len(vp), device=predictor.device).unsqueeze(1)
            transformed_points = predictor.transform.apply_coords_torch(input_points, img_array.shape[:2]).unsqueeze(1)
            masks, _, _ = predictor.predict_torch(
                point_coords=transformed_points,
                point_labels=input_labels,
                boxes=None,
                multimask_output=False,
            )
            for mask, color in zip(masks, mask_colors):
                img_mask = show_mask(img_mask, mask[0], color)
            for point, color in zip(input_points, mask_colors):
                img_mask = show_point(img_mask, point.cpu().numpy(), color)

        elif label == "Box":
            input_boxes = torch.tensor(vp, device=predictor.device)
            transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, img_array.shape[:2])
            masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
            for mask, color in zip(masks, mask_colors):
                img_mask = show_mask(img_mask, mask[0], color)
            for box, color in zip(input_boxes, mask_colors):
                img_mask = show_box(img_mask, box.cpu().numpy(), color)
    else:
        img_mask = img

    return img_mask


def reset(type):
    print("Resetting")
    chatbot = []
    chatbot_display = []
    if type == "Free-Form Draw":
        return None, None, json.dumps({"mask": []}), gr.update(visible=False), gr.update(visible=True), chatbot, chatbot_display, None, gr.update(visible=False)
    else:
        return None, None, json.dumps({"mask": []}), gr.update(visible=True), gr.update(visible=False), chatbot, chatbot_display, None, gr.update(visible=True)


def on_masks_change(type_selection: str, raw_img: Image, masks):
    masks_obj = json.loads(masks)["mask"]
    result = []
    for i, mask in enumerate(masks_obj):
        if len(mask) == 2:
            type = "Point %d" % i
        else:
            type = "Region %d" % i
        mask_str = str(tuple(mask))
        result.append([type, mask_str])
    out_img = draw_masks(type_selection, raw_img, masks_obj)
    return result, out_img


def upload_image(img):
    print("Image uploaded")
    return img


def undo(masks):
    masks_obj = json.loads(masks)["mask"]
    if len(masks_obj) >= 1:
        masks_obj = masks_obj[:-1]
    return json.dumps({"mask": masks_obj})


def show_user_input(msg, chatbot, chatbox_display):
    return "", chatbot + [[msg, None]], chatbox_display + [[msg, None]]


def gradio_worker(
    request_queues: List[mp.Queue], response_queue: mp.Queue,
    args: argparse.Namespace, barrier: mp.Barrier,
) -> None:
    """
    The gradio worker is responsible for displaying the WebUI and relay the
    requests to model workers. It should be launched only once.

    Args:
        request_queues (List[mp.Queue]): A list of request queues (one for
            each model worker).
        args (argparse.Namespace): All command line arguments.
        barrier (multiprocessing.Barrier): A barrier used to delay the start
            of Web UI to be after the start of the model.
    """

    sam = sam_model_registry["vit_h"](checkpoint="../checkpoints/sam/sam_vit_h_4b8939.pth").cuda()
    sam_predictor = SamPredictor(sam)

    def stream_model_output(img, masks, type_selection, img_input_draw, chatbot, chatbot_display, max_gen_len, gen_t, top_p, img_transform):
        if type_selection =="Free-Form Draw":
            vp = extract_masks_from_draw(img_input_draw)
            img = img_input_draw['background']
        else:
            if masks is not None:
                vp_js = json.loads(masks)
            else:
                vp_js = None
            vp = vp_js["mask"]

        # print("The coordinates for [x0,y0,x1,y1] of visual prompts are :" + str(vp))

        while True:
            content_piece = response_queue.get()
            if isinstance(content_piece, Ready):
                break

        for queue in request_queues:            
            queue.put((img, vp, chatbot, max_gen_len, gen_t, top_p, img_transform))
        while True:
            content_piece = response_queue.get()
            if isinstance(content_piece, ModelFailure):
                raise RuntimeError
            chatbot_display[-1][1] = content_piece['text'].replace("<", "&lt;").replace(">", "&gt;")
            if content_piece["end_of_content"]:
                chatbot[-1][1] = content_piece['text']
                question = content_piece['input']
                chatbot_display[-1][1] = content_piece['text']

                if img is not None:
                    masked_image = draw_mask_on_image(img, vp, sam_predictor)
                else:
                    masked_image = None
                yield chatbot, chatbot_display, masked_image
                break
    

    with gr.Blocks() as demo:
        gr.HTML("""
            <center> <h1>🎨 Draw-and-Understand: Leveraging Visual Prompts to Enable MLLMs to Comprehend What You Want</h1> </center>
            <center> <h2> SPHINX-V </h2> </center> <br/>
            <center>
                <a href="https://draw-and-understand.github.io/" style="margin-right: 25px; font-size: 20px;">🏠 [Project Page]</a>
                <a href="https://github.com/AFeng-x/Draw-and-Understand" style="margin-right: 25px; font-size: 20px;">💻 [Code]</a>
                <a href="" style="font-size: 20px;">📝 [Paper]</a> 
            </center>
            <br/>
            <div style="padding: 8px; border-radius: 10px; border: 2px solid red; max-width: 65%; margin: 0 auto;">
            <center> <p style="font-weight: bold; font-size:22px;">💡 Using Tips: </p> </center> 

            <p style="font-size:16px;">📌 First, pick a visual prompt type, then upload your image.</p>

            <p style="font-size:16px;">🔘 For the Point-type, click on the spot you're interested in.</p>

            <p style="font-size:16px;">🔲 For the Box-type, click the top-left and bottom-right corners to highlight your area of interest.</p>

            <p style="font-size:16px;">🖌️ For the Draw-type, one layer represents one area. To include multiple areas at once, simply click the "add layer" button at the bottom left.</p>

            <p style="font-size:16px;">⌛️ Due to network delays, it might take 1 ~ 3 seconds to see the visual prompts. Please wait a bit.</p>

            <p style="font-size:16px;">🆑 Click the clear button to create a new chat.</p>
            </div>

            """)

        with gr.Row():
            with gr.Column(variant="panel", scale=1):
                with gr.Row():
                    type_selection = gr.Radio(
                        choices=["Point", "Box", "Free-Form Draw"], value="Point", label="Type of Visual Prompt")
                with gr.Row():
                    raw_img = gr.Image(
                        label='Image Input', type='pil', visible=False)
                with gr.Row() as point_row:
                    img_input = gr.Image(
                        label='Image Input', type='pil', sources=['upload'])
                with gr.Row(visible=False) as draw_row:
                    img_input_draw = gr.ImageEditor(
                        label='Image Input',image_mode="RGB", type='pil', sources=['upload'], transforms=[], eraser=False, interactive=True)
                with gr.Row(visible=False):
                    masks = gr.Textbox(
                        interactive=False, value=json.dumps({"mask": []}))
                with gr.Row(visible=False):
                    mask_disp = gr.Dataframe(
                        interactive=False, headers=["Type", "Data"])
                with gr.Row():
                    undo_btn = gr.Button('🔙 Undo', variant="secondary")                    
            with gr.Column(variant="panel", scale=2):
                with gr.Row():
                    chat_history = gr.Chatbot(visible=False)
                    chat_history_display = gr.Chatbot(label="Chat History")
                with gr.Row():
                    prompt = gr.Textbox(lines=4, label="Question")
                with gr.Row():
                    with gr.Column():
                        run_botton = gr.Button("Run", variant='primary')
                        
                    with gr.Column():
                        clear_botton = gr.Button("Clear", variant='stop')

        with gr.Row():
            max_gen_len = gr.Slider(
                minimum=1, maximum=1024,
                value=1024, interactive=True,
                label="Single-turn max response length",
            )
            gen_t = gr.Slider(
                minimum=0, maximum=1, value=0.1, interactive=True,
                label="Temperature",
            )
            top_p = gr.Slider(
                minimum=0, maximum=1, value=0.75, interactive=True,
                label="Top-p",
            )
            img_transform = gr.Dropdown(choices=["padded_resize", "resized_center_crop"],
                                    value="padded_resize", label="Image Transform", visible=False)

        with gr.Row():
            with gr.Column(scale=1.5):
                with gr.Row():
                    output_image = gr.Image(interactive=False, label="Segmentation")
            with gr.Column(scale=2):
                with gr.Row():
                    gr.Markdown("""
                            <div style="padding: 8px; border-radius: 10px; border: 2px solid green;">
                            <p style="font-weight: bold; font-size:22px;">📖 Prompts: </p>

                            <p style="font-size:16px;"><b>• Category Identification:</b> Please identify the labels of each marked <span style="color: blue;">point</span>/<span style="color: red;">region</span> in the image.</p>

                            <p style="font-size:16px;"><b>• Text Recognion:</b> Please provide the ocr results of each marked <span style="color: blue;">point</span>/<span style="color: red;">region</span> in the image.</p>

                            <p style="font-size:16px;"><b>• Brief Caption:</b> Please provide a brief description of each marked <span style="color: blue;">point</span>/<span style="color: red;">region</span> in the image.</p>

                            <p style="font-size:16px;"><b>• Detailed Caption:</b> Please provide a detailed description of each marked <span style="color: blue;">point</span>/<span style="color: red;">region</span> in the image.</p>

                            <p style="font-size:16px;"><b>• Summarized Caption:</b> Please provide a summarized description based on all the marked <span style="color: blue;">point</span>/<span style="color: red;">region</span> in the image.</p>

                            <p style="font-size:16px;"><b>• Relationship Analyze:</b> Please analyze the relationship between all marked <span style="color: blue;">point</span>/<span style="color: red;">region</span> in the image.</p> 

                            <p style="font-size:16px;"><b>• Q&A:</b> Use the <span style="color: blue;">&lt;Mark n&gt;</span> or <span style="color: red;">&lt;Region n&gt;</span> identifiers to mark the pixel or region of interest to ask any questions.</p>
                            </div>
                            """,
                            label="Tips"
                            )


        masks.change(on_masks_change, inputs=[type_selection, raw_img, masks], outputs=[mask_disp, img_input])

        type_selection.change(reset, inputs=[type_selection],  outputs=[img_input, raw_img, masks, point_row, draw_row, chat_history, chat_history_display, output_image, undo_btn])

        img_input.upload(fn=upload_image,
                            inputs=[img_input], outputs=[raw_img])
        img_input.clear(fn = reset, inputs=[type_selection],  outputs=[img_input, raw_img, masks, point_row, draw_row, chat_history, chat_history_display, output_image, undo_btn])

        img_input_draw.clear(fn = reset, inputs=[type_selection],  outputs=[img_input, raw_img, masks, point_row, draw_row, chat_history, chat_history_display, output_image, undo_btn])

        img_input.select(fn=on_draw, inputs=[type_selection, masks], outputs=[masks])

        # img_input_draw.change(fn=draw_on_img, inputs=[type_selection, masks, img_input_draw], outputs=[masks])
        
        run_botton.click(
            show_user_input, [prompt, chat_history, chat_history_display], [prompt, chat_history, chat_history_display],
        ).then(
            stream_model_output, [raw_img, masks, type_selection, img_input_draw, chat_history, chat_history_display, max_gen_len, gen_t, top_p, img_transform],
            [chat_history, chat_history_display, output_image]
        )

        undo_btn.click(fn= undo, inputs=[masks], outputs=[masks])

        clear_botton.click(reset, inputs=[type_selection],  outputs=[img_input, raw_img, masks, point_row, draw_row, chat_history, chat_history_display, output_image, undo_btn])

    barrier.wait()
    demo.queue(api_open=True).launch(
        share=True,
        server_name="0.0.0.0", # "127.0.0.1"
    )


