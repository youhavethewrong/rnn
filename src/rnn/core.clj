(ns rnn.core)

(defn forward-multiply-gate
  [x y]
  (* x y))

(defn forward-add-gate
  [x y]
  (+ x y))

(defn forward-circuit
  [x y z]
  (forward-multiply-gate
   (forward-add-gate x y) z))

(defn random-tweak
  [f x y]
  (let [tweak-amount 0.01]
    (loop [best-result Integer/MIN_VALUE
           best-x x
           best-y y
           iterations 100]
      (if (= 0 iterations)
        {:result best-result :x best-x :y best-y}
        (let [try-x (+ x (* tweak-amount (dec (* (rand) 2))))
              try-y (+ y (* tweak-amount (dec (* (rand) 2))))
              try-result (forward-multiply-gate try-x try-y)
              best-result (if  (> try-result best-result) try-result best-result)
              best-x (if (= best-result try-result) try-x best-x)
              best-y (if (= best-result try-result) try-y best-y)]
          (recur best-result best-x best-y (dec iterations)))))))

(defn -derivative
  [f x y h]
  (/ (- (f (+ x h) y) (f x y)) h)
)

(defn numerical-gradient
  ([f x y h]
   [(/ (- (f (+ x h) y) (f x y)) h)
    (/ (- (f x (+ y h)) (f x y)) h)])
  
  ([f x y z h]
   [(/ (- (f (+ x h) y z) (f x y z)) h)
    (/ (- (f x (+ y h) z) (f x y z)) h)
    (/ (- (f x y (+ z h)) (f x y z)) h)]))

(defn numerical-derivative-tweak
  [f x y]
  (let [h 0.0001
        step-size 0.01
        x (+ x (* step-size (-derivative f x y h)))
        y (+ y (* step-size (-derivative f y x h)))]
    {:result (f x y) :x x :y x}))

(defn numerical-forward-circuit-gradient
  [x y z]
  (let [h 0.0001]
    (numerical-gradient forward-circuit x y z h)))

(defn analytical-forward-circuit-gradient
  [x y z]
  (let [q (forward-add-gate x y)
        f (forward-multiply-gate q z)
        der-f-wrt-z q
        der-f-wrt-q z
        der-q-wrt-x 1.0
        der-q-wrt-y 1.0
        der-f-wrt-x (* der-q-wrt-x der-f-wrt-q)
        der-f-wrt-y (* der-q-wrt-y der-f-wrt-q)]
    [der-f-wrt-x der-f-wrt-y der-f-wrt-z]))

(defn analytical-forward-circuit
  [x y z]
  (let [der-f-wrt-xyz (analytical-forward-circuit-gradient x y z)
        step-size 0.01
        x (+ x (* step-size (first der-f-wrt-xyz)))
        y (+ y (* step-size (second der-f-wrt-xyz)))
        z (+ z (* step-size (nth der-f-wrt-xyz 2)))]
    (forward-circuit x y z)))
