(* Simple library to do neural networks *)
open Printf

let sum = List.fold_left ( +. ) 0.

let sigmoid x = 1. /. (1. +. (exp (-.x)))

type matrix = { rows: int;
                cols: int;
                data: float array; }

let matrix_get m i j =
  assert (i < m.rows);
  (* if i >= m.rows || j >= m.cols then raise (Invalid_argument (sprintf "%d %d > %d %d" m.rows m.cols i j)); *)
  assert (j < m.cols);
  m.data.(i * m.cols + j)

let matrix_set m i j v =
  assert (i < m.rows);
  assert (j < m.cols);
  m.data.(i * m.cols + j) <- v

let matrix_inc m i j v =
  let p = matrix_get m i j in
  matrix_set m i j (p +. v)

let matrix_fill r c v =
  { rows = r;
    cols = c;
    data = Array.make (r*c) v }

let matrix_copy m =
  { rows = m.rows;
    cols = m.cols;
    data = Array.copy m.data }

let matrix_transpose m =
  let r = matrix_fill m.cols m.rows 0.0 in
  for i = 0 to m.rows-1 do
    for j = 0 to m.cols-1 do
      matrix_set r i j (matrix_get m j i);
    done
  done;
  r

let matrix_map fn m =
  { rows = m.rows;
    cols = m.cols;
    data = Array.map fn m.data }

let matrix_map2 fn a b =
  assert (a.rows == b.rows);
  assert (a.cols == b.cols);
  let data = Array.map2 fn a.data b.data in
  { rows = a.rows; cols = a.cols; data; }

let matrix_map3 fn a b c =
  assert (a.rows == b.rows);
  assert (a.cols == b.cols);
  assert (a.rows == c.rows);
  assert (a.cols == c.cols);
  let data = Array.make (a.rows * a.cols) 0. in
  for i=0 to (a.rows-1)*(a.cols-1) do
    data.(i) <- fn a.data.(i) b.data.(i) c.data.(i);
  done;
  { rows = a.rows; cols = a.cols; data = data; }

let matrix_add = matrix_map2 (+.)

let matrix_sub = matrix_map2 (-.)

let matrix_mul a b =
  assert (a.cols == b.rows);
  let m = matrix_fill a.rows b.cols 0.0 in
  for i = 0 to a.rows-1 do
    for j = 0 to b.cols-1 do
      for k = 0 to a.cols-1 do
        let v = (matrix_get m i j) +. (matrix_get a i k) *. (matrix_get b k j) in
        matrix_set m i j v
      done
    done
  done;
  m

let matrix_mul_sc s m =
  let mul = ( *. ) in
  matrix_map (mul s) m

let matrix_rand r c =
  let rand _ = Random.float 1.0 in
  { rows = r;
    cols = c;
    data = Array.init (r * c) rand }

let vector_dot a b =
  assert (a.cols == 1 && b.cols == 1);
  let s = ref 0. in
  for i = 0 to a.rows-1 do
    let c = (matrix_get a i 0) *. (matrix_get b i 0) in
    s := !s +. c;
  done;
  !s

let matrix_print m name =
  printf "%s = [\n" name;
  for i=0 to m.rows-1 do
    printf "\t";
    for j=0 to m.cols-1 do
      printf "%f " (matrix_get m i j);
    done;
    printf "\n";
  done;
  printf "]\n";

(* neural network structure:
   x1 --\
        C
   x2 --/
*)
type layer = { mutable w: matrix;
               mutable b: matrix;
               (** corresponds to previous layer's activation(current layer's input) *)
               mutable a: matrix; }

let layer_rand i o =
  { w = matrix_rand o i;
    b = matrix_rand o 1;
    a = matrix_rand o 1; }

let layer_copy l =
  { w = matrix_copy l.w;
    b = matrix_copy l.b;
    a = matrix_copy l.a; }

let layer_map2 fn a b = { w = matrix_map2 fn a.w b.w;
                          b = matrix_map2 fn a.b b.b;
                          a = matrix_map2 fn a.a b.a; }

(** sigma(w * x + b) *)
let layer_forward l x =
  l.a <- x;
  matrix_mul l.w x |>
  matrix_add l.b |>
  matrix_map sigmoid

type model = layer list

let model_rand i arch =
  let rec new_layer prev = function
    | [] -> []
    | n :: t ->
      let l = layer_rand prev n in
      [l] @ new_layer n t in
  new_layer i arch

let rec model_copy = function
  | [] -> []
  | h :: t ->
    let l = layer_copy h in
    [l] @ model_copy t

let rec model_forward x = function
  | [] -> x
  | h :: t ->
    let fwd = layer_forward h x in
    model_forward fwd t

let model_print m name =
  let rec print i = function
  | [] -> ()
  | h :: t ->
    matrix_print h.w (sprintf "w%d" i);
    matrix_print h.b (sprintf "b%d" i);
    print (i+1) t in
  printf "%s = [\n" name;
  print 0 m;
  printf "]\n"

let cost m x_train y_train =
  assert (List.length x_train == List.length y_train);
  let n = List.length x_train |> float_of_int in
  let error x y_train =
    let y = model_forward x m in
    let e = matrix_sub y y_train in
    vector_dot e e in
  (sum (List.map2 error x_train y_train)) /. n

let learn_finite_diff m x_train y_train =
  let eps = 0.1 and rate = 0.1 in
  let s = cost m x_train y_train in
  let d u x = u -. (x -. s) /. eps *. rate in

  let tmp = model_copy m in

  let wiggle_w gl ml i j =
    let p = matrix_get gl.w j i in
    matrix_set gl.w j i (p +. eps);
    let c = cost tmp x_train y_train |> d p in
    matrix_set ml.w j i c;
    matrix_set gl.w j i p in
  let wiggle_b gl ml i =
    let p = matrix_get gl.b 0 i in
    matrix_set gl.b 0 i (p +. eps);
    let c = cost tmp x_train y_train |> d p in
    matrix_set ml.b 0 i c;
    matrix_set gl.b 0 i p in
  let wiggle_layer gl =
    let ml = layer_copy gl in
    for i=0 to gl.b.cols-1 do
      wiggle_b gl ml i;
    done;
    for i=0 to gl.w.cols-1 do
      for j=0 to gl.w.rows-1 do
        wiggle_w gl ml i j;
      done
    done;
    ml in

  List.map wiggle_layer tmp

(** Returns a new model with initial state [m] trained on [x_train]
    and [y_train].  [rate] specifies how much model will be
    trained. NOTE: it's better to [learn_backprop] multiple times
    instead of passing a big rate. [1.0] is just okay for [rate]. *)
let learn_backprop m x_train y_train rate =
  assert (List.length x_train == List.length y_train);
  let n_inv = rate /. (List.length x_train |> float_of_int) in
  (* compute gradients *)
  let per_sample x_sample y_sample =
    let ith mat i = matrix_get mat i 0 in
    let y_pred = model_forward x_sample m in
    (* gradient of last layer *)
    let g = matrix_map2 (fun a b -> (a -. b)) y_pred y_sample in
    let rec propagate a da acc = function
      | [] -> acc
      | layer :: t ->
        let ga = matrix_fill (layer.a.rows) 1 0.
        and gw = matrix_fill (layer.w.rows) (layer.w.cols) 0. in
        let gb = matrix_map2 (fun a da -> 2. *. a *. da *. (1. -. a)) a da in
        for i=0 to layer.w.rows-1 do
          let d = (ith a i) *. (ith da i) *. (1. -. (ith a i)) in
          for j=0 to layer.w.cols-1 do
            let pa = matrix_get layer.a j 0
            and w = matrix_get layer.w i j in
            matrix_inc gw i j (2. *. d *. pa);
            matrix_inc ga j 0 (4. *. d *. w);
          done
        done;
        let ga = matrix_mul_sc n_inv ga
        and gw = matrix_mul_sc n_inv gw
        and gb = matrix_mul_sc n_inv gb in
        (* go to previous layer *)
        let gl = { w = gw; b = gb; a = ga; } in
        propagate layer.a ga ([gl] @ acc) t
    in propagate y_pred g [] (List.rev m) in
  (* add gradients *)
  let add_models = List.map2 (layer_map2 (+.)) in
  let sub_models = List.map2 (layer_map2 (-.)) in
  let grad = match List.map2 per_sample x_train y_train with
    | [] -> failwith "internal error"
    | h :: t -> List.fold_left add_models h t in
  (* gradient descent *)
  sub_models m grad

(** Teach model [m] for [n] iterations with finite differences algorithm.
    [x_train] and [y_train] are data which will teach the model. *)
let rec teach_fd x_train y_train n m =
  let m = learn_finite_diff m x_train y_train in
  if n > 0 then teach_fd x_train y_train (n-1) m
  else m, (cost m x_train y_train)

(** Teach model [m] for [n] iterations with finite back propagation algorithm.
    [x_train] and [y_train] are data which will teach the model. *)
let rec teach_bp x_train y_train n m =
  let m = learn_backprop m x_train y_train 1. in
  if n > 0 then teach_bp x_train y_train (n-1) m
  else m, (cost m x_train y_train)

(** prepares dataset as a tuple of matrices *)
let prepare_train x_count y_count train =
  let rec impl xacc yacc = function
  | [] -> xacc, yacc
  | h :: t ->
    let x = { rows = x_count; cols = 1; data = Array.sub h 0 x_count; }
    and y = { rows = y_count; cols = 1; data = Array.sub h x_count y_count; } in
    impl (x :: xacc) (y :: yacc) t
  in impl [] [] train

type image = Stb_image.int8 Stb_image.t

let image_load path =
  match Stb_image.load path with
    | Error message ->
      (match message with
       | `Msg str ->
         sprintf "failed to load image with message %s\n" str |> failwith)
    | Ok image ->
      image

(** NOTE: png is saved *)
let image_save path width height channels data =
  Stb_image_write.png path ~w:width ~h:height ~c:channels data

(** Convert image to tuple of lists of matrices. *)
let image_to_train (img: image) =
  let width = img.width |> float_of_int
  and height = img.height |> float_of_int in
  let rec per_pixel x y xacc yacc =
    if y >= img.height then xacc, yacc
    else if x >= img.width then per_pixel 0 (y+1) xacc yacc
    else begin
    let pixel = (x + y * img.width) * img.channels in
    let input = matrix_fill 2 1 0.
    and output = matrix_fill 3 1 0. in
    matrix_set input 0 0 ((x |> float_of_int) /. width);
    matrix_set input 1 0 ((y |> float_of_int) /. height);
    matrix_set output 0 0 ((img.data.{pixel} |> float_of_int) /. 255.);
    matrix_set output 1 0 ((img.data.{pixel + 1} |> float_of_int) /. 255.);
    matrix_set output 2 0 ((img.data.{pixel + 2} |> float_of_int) /. 255.);
    (* printf "%f %f %f\n" (matrix_get output 0 0) (matrix_get output 1 0) (matrix_get output 2 0); *)
    per_pixel (x+1) y (input :: xacc) (output :: yacc)
    end
  in per_pixel 0 0 [] []

let model_save_image path width height model =
  let w = width |> float_of_int and h = height |> float_of_int in
  let channels = 3 in
  let buffer = Bigarray.Array1.create Bigarray.int8_unsigned Bigarray.c_layout (width * height * channels) in
  for y=0 to height-1 do
    for x=0 to width-1 do
      let input = matrix_fill 2 1 0. in
      matrix_set input 0 0 ((float_of_int x) /. w);
      matrix_set input 1 0 ((float_of_int y) /. h);
      let output = model_forward input model in
      let pixel = (x + y * width) * channels in
      buffer.{pixel} <- ((matrix_get output 0 0) *. 255.) |> int_of_float;
      buffer.{pixel+1} <- ((matrix_get output 1 0) *. 255.) |> int_of_float;
      buffer.{pixel+2} <- ((matrix_get output 2 0) *. 255.) |> int_of_float;
      (* printf "%d %d -> %d %d %d\n" x y buffer.{pixel} buffer.{pixel+1} buffer.{pixel+2}; *)
    done;
  done;
  image_save path width height channels buffer

let shuffle_list d =
  let nd = List.map (fun c -> (Random.bits (), c)) d in
  let sond = List.sort compare nd in
  List.map snd sond

let shuffle_train x_train y_train =
  let rec to_list x y acc =
    match x with
    | [] -> acc
    | xh :: xt ->
      match y with
      | [] -> failwith "internal error"
      | yh :: yt ->
        let el = xh, yh in
        to_list xt yt (el :: acc) in
  let rec from_list xacc yacc = function
    | [] -> xacc, yacc
    | h :: t -> let x, y = h in
      from_list (x :: xacc) (y :: yacc) t in
  let l = to_list x_train y_train [] |> shuffle_list in
  from_list [] [] l

let teach_fast x_train y_train chunk_size n m =
  let x_train, y_train = shuffle_train x_train y_train in
  let size = List.length x_train in
  let num_chunks = size / chunk_size in
  let x_chunks = Array.make num_chunks [] in
  let y_chunks = Array.make num_chunks [] in
  let rec fill_chunks i x y =
    if i == size then ()
    else begin
      x_chunks.(i mod num_chunks) <- x_chunks.(i mod num_chunks) @ [List.hd x];
      y_chunks.(i mod num_chunks) <- y_chunks.(i mod num_chunks) @ [List.hd y];
      fill_chunks (i+1) (List.tl x) (List.tl y)
    end in
  fill_chunks 0 x_train y_train;
  let rec teach i m =
    if i >= n then m
    else
      let x = x_chunks.(i mod num_chunks) and y = y_chunks.(i mod num_chunks) in
      if i mod (n / 10) == 0 then begin
        printf "Teaching: %d0%%, %dth iteration, cost=%f\n" (i / (n / 10)) i (cost m x_train y_train);
        flush_all ();
      end;
      teach (i+1) (learn_backprop m x y 1.) in
  let m = teach 0 m in
  m, cost m x_train y_train
