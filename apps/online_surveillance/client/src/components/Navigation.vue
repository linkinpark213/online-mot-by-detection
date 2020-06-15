<template>
  <div id="navigation">
    <div id="logo">
      <img src="../assets/logo.png" alt="logo" width="64" height="64">
    </div>
    <div id="site-name">
      {{ siteName }}
    </div>
    <a class="nav-button" v-for="button in buttons" href="#" v-bind:key="button.name">
      {{ button.name }}
    </a>
  </div>
</template>

<script>
export default {
  name: 'Navigation',
  data() {
    return {
      siteName: 'MTMCT Client',
      buttons: [
        {
          name: 'Add'
        }, {
          name: 'Remove'
        }
      ]
    }
  }
}
</script>

<style>
  #navigation {
    background-color: #455A64;
    color: white;
    line-height: 64px;
    height: 64px;
    text-align: left;
    display: block;
    width: 100%;
  }

  a {
    text-decoration: none;
    color: white;
  }

  #logo {
    display: inline;
    float: left;
    margin: 0 10px 0 10px;
  }

  #site-name {
    display: inline;
    margin: 0 10px 0 0;
    height: 100%;
    line-height: 100%;
    font-weight: bold;
  }

  .nav-button {
    padding: 0 10px 0 10px;
    display: inline-block;
    transition: all ease 200ms;
    height: 100%;
  }

  .nav-button:hover {
    background-color: #42b983;
  }
</style>
