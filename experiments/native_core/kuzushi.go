package main

/*
#include <stdlib.h>
*/
import "C"
import (
	"math"
	"unsafe"
)

//export FastKuzushi
// FastKuzushi takes a pointer to a flattened 17x3 float64 array (kpts),
// calculates the Kuzushi metrics, populates outputs, and returns 1 if Kuzushi is detected, else 0.
func FastKuzushi(kptsPtr *C.double, outDist *C.double, outComX *C.double, outComY *C.double) C.int {
	// Cast the C pointer to a Go array pointer to access elements directly
	kpts := (*[51]float64)(unsafe.Pointer(kptsPtr))

	// Hips (11, 12)
	p11x, p11y := kpts[11*3+0], kpts[11*3+1]
	p12x, p12y := kpts[12*3+0], kpts[12*3+1]
	pelvisX := (p11x + p12x) / 2.0
	pelvisY := (p11y + p12y) / 2.0

	// Shoulders (5, 6)
	p5x, p5y := kpts[5*3+0], kpts[5*3+1]
	p6x, p6y := kpts[6*3+0], kpts[6*3+1]
	neckX := (p5x + p6x) / 2.0
	neckY := (p5y + p6y) / 2.0

	// Ankles (15, 16) Left is 15, Right is 16
	p15x, p15y := kpts[15*3+0], kpts[15*3+1]
	p16x, p16y := kpts[16*3+0], kpts[16*3+1]

	// Center of Mass (Weighted 60% lower, 40% upper)
	comX := (pelvisX * 0.6) + (neckX * 0.4)
	comY := (pelvisY * 0.6) + (neckY * 0.4)

	*outComX = C.double(comX)
	*outComY = C.double(comY)

	// Stance width
	dx := p15x - p16x
	dy := p15y - p16y
	stanceWidth := math.Sqrt(dx*dx + dy*dy)

	// Vector AB (r_ankle to l_ankle)
	abX, abY := dx, dy // Because l_ankle(15) - r_ankle(16)
	apX, apY := comX-p16x, comY-p16y

	dotAB := abX*abX + abY*abY
	var t float64 = 0.0
	if dotAB != 0 {
		t = (apX*abX + apY*abY) / dotAB
		if t < 0.0 {
			t = 0.0
		} else if t > 1.0 {
			t = 1.0
		}
	}

	// Closest point
	cpX := p16x + t*abX
	cpY := p16y + t*abY

	// Distance to base
	distX := comX - cpX
	distY := comY - cpY
	dist := math.Sqrt(distX*distX + distY*distY)

	*outDist = C.double(dist)

	// Threshold evaluation
	dynamicThreshold := 15.0 + (stanceWidth * 0.30)
	if dist > dynamicThreshold {
		return C.int(1)
	}
	return C.int(0)
}

func main() {}
