open Nn
open Printf

let main input_path output_path epochs w_rate h_rate m =
  let img = image_load input_path in
  printf "gathering data...\n";
  let x_train, y_train = image_to_train img in
  printf "starting training process...\n";
  flush_all ();
  (* let m_trained, c = teach_fast x_train y_train 32 epochs m in *)
  let m_trained, c = teach_bp x_train y_train epochs m in
  let width  = w_rate *. (float_of_int img.width) |> int_of_float
  and height = h_rate *. (float_of_int img.height) |> int_of_float in
  model_save_image output_path width height m_trained;
  printf "Cost: %f\n" c;
  m_trained

;;

Random.init 69;;
let m = model_rand 2 [12; 12; 3];;

(*
let m = model_read_from_file "end.json";;

let it =
  if Array.length Sys.argv < 2 then 100
  else Sys.argv.(1) |> int_of_string in
(* main "./pictures/cough_cat.png" "./pictures/result_cough_cat.png" (it*1000) 2. 2. m *)
main "./pictures/cough_cat.png" "./pictures/result_cough_cat.png" it 2. 2. m
|> model_save_to_file "end.json"
*)
Nn_ui.init_gui ();;

ignore main;;
(* let model_widget = Nn_ui.make_widget 0 0 500 300 (Nn_ui.draw_model m) in *)
let result_widget = Nn_ui.make_widget 500 200 20 20 Nn_ui.draw_model_as_image in

let path =
  if Array.length Sys.argv < 2 then "pictures/dataset4.png"
  else Sys.argv.(1) in
let img = image_load path in
let x_train, y_train = image_to_train img in
try Nn_ui.main_loop m 0 x_train y_train [result_widget]
with Exit -> ();;

Nn_ui.free_gui ();;
