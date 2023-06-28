open Nn
open Printf

let main input_path output_path epochs w_rate h_rate m =
  let img = image_load input_path in
  printf "gathering data...\n";
  let x_train, y_train = image_to_train img in
  printf "starting training process...\n";
  flush_all ();
  let m_trained, c = teach_fast x_train y_train 32 epochs m in
  let width  = w_rate *. (float_of_int img.width) |> int_of_float
  and height = h_rate *. (float_of_int img.height) |> int_of_float in
  model_save_image output_path width height m_trained;
  printf "Cost: %f\n" c

;;

Random.init 69;;
let m = model_rand 2 [8; 8; 3];;

let it =
  if Array.length Sys.argv < 2 then 100
  else Sys.argv.(1) |> int_of_string in
main "./pictures/dataset4.png" "./pictures/result4.png" (it*1000) 3. 2.5 m
