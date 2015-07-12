(ns rnn.core-test
  (:require [clojure.test :refer :all]
            [rnn.math :refer [double=]]
            [rnn.core :refer :all]))

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
