open Graphics
open Nn
open Printf

let init_gui () =
  open_graph " 640x480";
  auto_synchronize false;
  (* set_text_size doesn't work apparently, that's a shame. I might as
     well rewrite everything in a better language, or even use
     Python. *)
  set_text_size 40
  (* set_font "-*-fixed-medium-r-semicondensed--50-*-*-*-*-*-iso8859-1" *)

let free_gui () =
  close_graph ()

type widget = {
  x: int;
  y: int;
  w: int;
  h: int;
  draw: model -> int -> int -> int -> int -> unit;
}

let make_widget x y w h f =
  { x; y; w; h; draw = f; }

let draw_widget m w =
  w.draw m w.x w.y w.w w.h

let rec main_loop m epoch x_train y_train widgets =
  (if key_pressed () then
     match read_key () with
     | 'q' -> raise Exit
     | _ -> ()
  );

  synchronize ();
  clear_graph ();
  set_color (rgb 190 220 0);

  let de = 50 in
  let m_trained, c = teach_bp x_train y_train de m in

  moveto 100 600;
  set_color (rgb 0 0 0);
  let str = sprintf "Cost: %f, Epoch: %d" c epoch in
  draw_string str;

  List.map (draw_widget m_trained) widgets |> ignore;


  main_loop m_trained (epoch+de) x_train y_train widgets

let draw_model m x y w h =
  let int_x fx = int_of_float(fx *. float_of_int(w)) + x
  and int_y fy = int_of_float(fy *. float_of_int(h)) + y
  and int_s sz = sz *. float_of_int(min w h) |> int_of_float in
  let fill_circle x y r = fill_circle (int_x x) (int_y y) (int_s r)
  in
  let hpad = 0.2 and vpad = 0.2 in
  let color t =
    (* low color *)
    let lr = 0. and lg = 0. and lb = 1.
    (* high color *)
    and hr = 1. and hg = 0. and hb = 0. in
    (* value range *)
    let l = -1. and r = 1. in
    let norm x = (x -. l) /. (r -. l) in
    let to_255 x = x *. 255. |> int_of_float in
    let r = norm t *. (hr -. lr) +. lr |> to_255
    and g = norm t *. (hg -. lg) +. lg |> to_255
    and b = norm t *. (hb -. lb) +. lb |> to_255 in
    rgb r g b in
  (* calculate radius for circles *)
  let num = List.fold_left (fun a l -> max a l.b.rows) 0 m in
  let r = (1. -. 2. *. vpad) *. 0.5 /. float_of_int num in
  (* draw input layer *)
  let first_layer = List.hd m in
  set_color (rgb 127 127 127);
  for i=0 to first_layer.w.cols-1 do
    let y0 = (1. -. (float_of_int first_layer.w.cols) *. r) *. 0.5 in
    fill_circle hpad (y0 +. (float_of_int i) /. float_of_int first_layer.w.cols) r
  done;
  (* draw layers *)
  let draw_layer layer x =
    let neurons = layer.b.rows in
    let inv_n = 1. /. (float_of_int neurons) in
    let y0 = (1. -. (float_of_int neurons) *. r) *. 0.5 in
    let y i = (float_of_int i) *. inv_n in
    for i=0 to neurons-1 do
      let bias = matrix_get layer.b i 0 in
      set_color (color bias);
      fill_circle x (y0 +. (y i)) r
    done in
  let rec per_layer x = function
    | [] -> ()
    | h :: t ->
      draw_layer h x;
      per_layer (x +. r +. hpad) t in
  per_layer (2. *. hpad +. r) m

let draw_model_as_image m x y w h =
  let tou8 x = x *. 255. |> int_of_float in
  let wf = float_of_int w and hf = float_of_int h in
  let data = Array.init h (fun _ -> Array.make w 0) in
  for i=0 to h-1 do
    for j=0 to w-1 do
      let input = matrix_fill 2 1 0. in
      matrix_set input 0 0 ((float_of_int j) /. wf);
      matrix_set input 1 0 ((float_of_int i) /. hf);
      let y = model_forward input m in
      let color = (rgb (matrix_get y 0 0 |> tou8) (matrix_get y 1 0 |> tou8) (matrix_get y 2 0 |> tou8)) in
      data.(i).(j) <- color;
    done;
  done;
  let texture = make_image data in
  draw_image texture x y
