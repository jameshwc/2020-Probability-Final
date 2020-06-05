package main

import (
	"bufio"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"sort"
	"strings"
	"time"

	mapset "github.com/deckarep/golang-set"
)

const (
	filename   = "t1-data.csv"
	batchNum   = 100
	batchSize  = 100
	initScore  = .08
	sampleSize = 20000
	dataSize   = 2000000
)

func hellingerDistance(normP []float64, normQ []float64) float64 {
	s := 0.0
	for i := range normP {
		s += math.Sqrt(normP[i] * normQ[i])
	}
	return math.Sqrt(1 - s)
}
func parse(isFull bool, index []int) []map[string]int {
	file, err := os.Open(filename)
	if err != nil {
		log.Fatalf("read %s", filename)
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	scanner.Split(bufio.ScanLines)
	scanner.Scan()
	header := strings.Split(scanner.Text(), ",")
	dat := make([]map[string]int, len(header)-1)
	for i := range dat {
		dat[i] = make(map[string]int)
	}
	updateDat := func(text string) {
		row := strings.Split(text, ",")[1:]
		for idx, r := range row {
			dat[idx][r]++
		}
	}
	if isFull {
		for scanner.Scan() {
			updateDat(scanner.Text())
		}
	} else {
		for i, j := 0, 0; scanner.Scan() && j < len(index); i++ {
			if i == index[j] {
				updateDat(scanner.Text())
				j++
			}
		}
	}
	return dat
}

func compare(fullCD []map[string]int, sampleCD []map[string]int) float64 {
	var distList []float64
	for id := range fullCD {
		var normP, normQ []float64
		pSum, qSum := 0, 0
		for k := range fullCD[id] {
			pSum += fullCD[id][k]
			qSum += sampleCD[id][k]
		}
		for k := range fullCD[id] {
			normP = append(normP, float64(fullCD[id][k])/float64(pSum))
			normQ = append(normQ, float64(sampleCD[id][k])/float64(qSum))
		}
		distList = append(distList, hellingerDistance(normP, normQ))
	}
	s := 0.0
	for i := range distList {
		s += distList[i]
	}
	return s / float64(len(distList))
}

func greedy() {
	fullCD := parse(true, nil)
	sampleSet := mapset.NewSet()
	for sampleSet.Cardinality() < sampleSize {
		bestScore := initScore
		var bestBatch mapset.Set
		for i := 0; i < batchNum; i++ {
			batch := mapset.NewSet()
			for batch.Cardinality() < batchSize {
				r := rand.Intn(dataSize)
				for sampleSet.Contains(r) {
					r = rand.Intn(dataSize)
				}
				batch.Add(r)
			}
			newSampleSet := sampleSet.Union(batch).ToSlice()
			index := make([]int, len(newSampleSet))
			for i := 0; i < len(newSampleSet); i++ {
				index[i] = newSampleSet[i].(int)
			}
			sort.Ints(index)
			sampleCD := parse(false, index)
			if curScore := compare(fullCD, sampleCD); curScore < bestScore {
				bestBatch = batch
				bestScore = curScore
				fmt.Println(bestScore)
			}
		}
		sampleSet = sampleSet.Union(bestBatch)
		fmt.Println(sampleSet.Cardinality())
	}
	index := make([]int, sampleSize)
	sampleSlice := sampleSet.ToSlice()
	for i := 0; i < sampleSize; i++ {
		index[i] = sampleSlice[i].(int)
	}
	sort.Ints(index)
	fmt.Print(compare(fullCD, parse(false, index)))
}
func main() {
	rand.Seed(time.Now().Unix())
	greedy()
}
