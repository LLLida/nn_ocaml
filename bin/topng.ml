(* Converts an image of any format to png.
   Usage: topng <input.some_extension> [<output.png>]
   *)
open Nn

let input_path = Sys.argv.(1)
let output_path =
  if Array.length Sys.argv > 2 then Sys.argv.(2)
  else input_path
let image = image_load input_path

;;

image_save output_path image.width image.height image.channels image.data
