<template>
  <div id="main-container">
    <div id="main-container-wrapper">
      <Camera v-for="camera in cameras" v-bind:id="camera.id" v-bind:key="camera.id"/>
    </div>
  </div>
</template>

<script>
import Camera from './Camera'

export default {
  el: '#main-container',
  components: {Camera},
  data() {
    return {
      cameras: [
        {
          id: '1',
          imgUrl: 'https://192.168.1.105:8080/frame',
          targets: []
        }, {
          id: '2',
          imgUrl: 'https://192.168.1.106:8080/frame',
          targets: []
        }
      ]
    }
  }
}
</script>

<style>
  #main-container {
    text-align: center;
  }

  #main-container-wrapper {
    margin: 20px 10% 20px 10%;
  }
</style>
