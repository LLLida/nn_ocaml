open Nn;;
open Printf;;

Random.init 69;;
let m = model_rand 2 [8; 8; 3];;

let img = image_load "./pictures/dataset5.png" in
printf "gathering data...\n";
let x_train, y_train = image_to_train img in
(* let x_train, y_train = shuffle_train x_train y_train in *)
printf "starting training process...\n";
flush_all ();
let it =
  if Array.length Sys.argv < 2 then 100
  else Sys.argv.(1) |> int_of_string in
let m_trained, c = teach_fast x_train y_train 32 (it*1000) m in
(* let m_trained, c = teach_bp x_train y_train (10*1000) m in *)
model_save_image "./pictures/result5.png" (3*img.width) (2*img.height) m_trained;
printf "Cost: %f\n" c
