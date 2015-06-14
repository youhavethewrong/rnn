(ns rnn.core)

(defn forward-multiplier-gate
  [x y]
  (* x y))

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
              try-result (forward-multiplier-gate try-x try-y)
              best-result (if  (> try-result best-result) try-result best-result)
              best-x (if (= best-result try-result) try-x best-x)
              best-y (if (= best-result try-result) try-y best-y)]
          (recur best-result best-x best-y (dec iterations)))))))

(defn -derivative
  [f x y h]
  (/ (- (f (+ x h) y) (f x y)) h))

(defn derivative-tweak
  [f x y]
  (let [h 0.0001
        step-size 0.01
        x (+ x (* step-size (-derivative f x y h)))
        y (+ y (* step-size (-derivative f y x h)))]
    {:result (f x y) :x x :y x}))
