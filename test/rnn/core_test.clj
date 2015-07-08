(ns rnn.core-test
  (:require [clojure.test :refer :all]
            [rnn.core :refer :all]))

(defn- scale [x y]
  (if (or (zero? x) (zero? y))
    1
    (Math/abs x)))

(defn double=
  ([x y]
   (double= x y 0.00001))
  ([x y epsilon]
   (<= (Math/abs (- x y))
       (* (scale x y) epsilon))))

(deftest gate-tests
  (testing "Testing forward multiply gate."
    (is (= -6 (forward-multiply-gate -2 3))))
  (testing "Try to get something better than -6 by randomly tweaking the inputs."
    (is (> (:result (random-tweak forward-multiply-gate -2 3))
           -6)))
  (testing "Try to get something better than -6 using derivatives"
    (is (> (:result (numerical-derivative-tweak forward-multiply-gate -2 3))
           -6)))
  (testing "Testing forward circuit"
    (is (= (forward-circuit -2 5 -4)
           -12)))
  (testing "optimize with analytical solution"
    (is (> (analytical-forward-circuit -2 5 -4)
           -12)))
  (testing "check analytic solutions with numerical"
    (is (every? true? (map double= (analytical-forward-circuit-gradient -2 5 -4) (numerical-forward-circuit-gradient -2 5 -4)))))
  )
