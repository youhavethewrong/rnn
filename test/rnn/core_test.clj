(ns rnn.core-test
  (:require [clojure.test :refer :all]
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
  (testing "optimize "
    (is (> (analytical-forward-circuit -2 5 -4)
           -12))))
