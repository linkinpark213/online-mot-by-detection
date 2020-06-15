<template>
  <div class="camera">
    <div class="camera-title">Camera #{{id}}</div>
    <div class="graphics-interface">
      <div class="img-frame">
        <img :src="imgUrl" alt="Unable to fetch frame from camera.">
      </div>
      <div class="targets">
        <div class="targets-title">
          Tracked targets
        </div>
        <div class="targets-list">
          No targets tracked.
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  el: '#camera',
  name: 'Camera',
  props: ['id', 'imgUrl', 'targets']
}
</script>

<style>
  .camera {
    text-align: center;
  }

  .camera-title {
    font-size: larger;
    font-weight: bold;
    text-align: left;
  }

  .img-frame {
    width: 70%;
    height: 100%;
    display: inline-block;
    text-align: center;
    float: left;
    background-color: #EEEEEE;
  }

  .targets {
    width: 30%;
    display: inline-block;
    float: left;
    overflow: auto;
    text-align: center;
    background-color: #DDDDDD;
  }

  .targets-title {
    text-align: left;
    padding: 10px;
  }

  .targets-list {
    padding: 5%;
  }
</style>
